"""解混服务层测试：矩阵校验、图像约束、NNLS 正确性、饱和规则与产物。"""
from __future__ import annotations

import io

import numpy as np
import pytest
import tifffile

from app.errors import ConflictError, NotFoundError, UnmixingError
from app.unmix_service import UnmixingService

from .conftest import complete_upload_with_blob, make_cyx_ome_tiff, split_parts

# 响应矩阵：行=探测器，列=染料（染料0串色到探测器1，染料1串色到探测器0）
MATRIX = [[1.0, 0.25], [0.4, 1.0]]
DYES = ["DAPI", "Cy3"]


@pytest.fixture
def unmix(service):
    return UnmixingService(service.store)


def _ingest(service, tmp_path, **kwargs):
    blob = make_cyx_ome_tiff(tmp_path / "cyx.ome.tiff", **kwargs)
    return complete_upload_with_blob(service, service.store, "U1", "cyx.ome.tiff", blob)


# --------------------------------------------------------------- 成功路径
def test_unmix_happy_path(service, unmix, tmp_path, fake_s3):
    upload = _ingest(service, tmp_path, width=40, height=30)
    record = unmix.create_job(upload["upload_id"], DYES, MATRIX)

    assert record["status"] == "completed"
    assert record["job_id"] and record["result_object_key"]
    assert record["dye_names"] == DYES
    assert record["response_matrix"] == MATRIX
    assert record["width"] == 40 and record["height"] == 30
    assert record["saturated_pixels"] == 0
    assert record["valid_pixels"] == 40 * 30

    # 结果对象在湖中可下载，且为两通道 float32 CYX OME-TIFF
    _, stream = unmix.open_result_stream(record["job_id"])
    blob = stream.read()
    assert len(blob) == record["result_bytes"]
    with tifffile.TiffFile(io.BytesIO(blob)) as tif:
        result = tif.asarray()
        assert result.dtype == np.float32
        assert result.shape == (2, 30, 40)
        assert tif.series[0].axes == "CYX"
        names = tif.ome_metadata
        assert 'Name="DAPI"' in names and 'Name="Cy3"' in names

    # 原件保持不变（对象键不同、字节未被触碰）
    _, orig_stream = service.open_object_stream(upload["upload_id"])
    original = orig_stream.read()
    assert record["result_object_key"] != upload["object_key"]
    assert fake_s3.head_object(upload["object_key"])["size"] == len(original)


def test_nnls_recovers_known_dye_intensities(service, unmix, tmp_path):
    """用 A·x 合成读数，解混应在非负区域还原 x（残差为 0）。"""
    A = np.array(MATRIX)
    height, width = 8, 8
    # 构造真实染料强度（非负），混合成两路 uint16 读数
    rng = np.random.default_rng(1)
    true_x = rng.integers(0, 2000, size=(2, height, width)).astype(np.float64)
    mixed = np.einsum("ij,jyx->iyx", A, true_x)
    mixed = np.clip(np.rint(mixed), 0, 65534).astype(np.uint16)

    path = tmp_path / "synth.ome.tiff"
    with tifffile.TiffWriter(path) as w:
        w.write(mixed, photometric="minisblack", metadata={"axes": "CYX"})
    upload = complete_upload_with_blob(
        service, service.store, "U2", "synth.ome.tiff", path.read_bytes())

    record = unmix.create_job(upload["upload_id"], DYES, MATRIX)
    _, stream = unmix.open_result_stream(record["job_id"])
    with tifffile.TiffFile(io.BytesIO(stream.read())) as tif:
        result = tif.asarray()

    # 舍入误差应在 ±1 个原始计数以内
    assert np.all(np.isfinite(result))
    assert (result >= 0).all()
    assert np.allclose(result, true_x, atol=2.0)


def test_saturated_pixels_become_nan_both_channels(service, unmix, tmp_path):
    upload = _ingest(service, tmp_path, width=16, height=12, saturated=True)
    record = unmix.create_job(upload["upload_id"], DYES, MATRIX)
    assert record["status"] == "completed"
    assert record["saturated_pixels"] == 2
    assert record["valid_pixels"] == 16 * 12 - 2

    _, stream = unmix.open_result_stream(record["job_id"])
    with tifffile.TiffFile(io.BytesIO(stream.read())) as tif:
        result = tif.asarray()
    # 两个饱和像素的两路结果都是 NaN，其余像素有限且非负
    assert np.all(np.isnan(result[:, 0, 0]))
    assert np.all(np.isnan(result[:, 2, 3]))
    finite_mask = np.isfinite(result[0])
    assert finite_mask.sum() == record["valid_pixels"]
    assert (result[:, finite_mask] >= 0).all()


# --------------------------------------------------------------- 矩阵校验
def test_non_finite_matrix_rejected(service, unmix, tmp_path):
    upload = _ingest(service, tmp_path)
    for bad in [[[1.0, float("nan")], [0.0, 1.0]],
                [[1.0, float("inf")], [0.0, 1.0]]]:
        with pytest.raises(UnmixingError, match="非有限"):
            unmix.create_job(upload["upload_id"], DYES, bad)


def test_ill_conditioned_matrix_rejected(service, unmix, tmp_path):
    upload = _ingest(service, tmp_path)
    nearly_singular = [[1.0, 1.0], [0.0, 0.001]]  # 条件数 2000 > 1000
    with pytest.raises(UnmixingError, match="条件数"):
        unmix.create_job(upload["upload_id"], DYES, nearly_singular)


def test_wrong_matrix_shape_rejected(service, unmix, tmp_path):
    upload = _ingest(service, tmp_path)
    with pytest.raises(UnmixingError, match="2×2"):
        unmix.create_job(upload["upload_id"], DYES, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(UnmixingError):
        unmix.create_job(upload["upload_id"], DYES, [[1.0], [0.0]])


# --------------------------------------------------------------- 图像约束
def test_reject_non_cyx_axes(service, unmix, tmp_path):
    upload = _ingest(service, tmp_path, axes="ZYX", channels=3)
    with pytest.raises(UnmixingError, match="CYX"):
        unmix.create_job(upload["upload_id"], DYES, MATRIX)


def test_reject_four_dimensional_tcyx(service, unmix, tmp_path):
    import tifffile as tf
    path = tmp_path / "tcyx.ome.tiff"
    # T 维取 2，避免 tifffile 把大小为 1 的维度折叠掉
    with tf.TiffWriter(path, ome=True) as w:
        w.write(np.zeros((2, 2, 8, 8), dtype=np.uint16),
                metadata={"axes": "TCYX"})
    upload = complete_upload_with_blob(
        service, service.store, "TC", "tcyx.ome.tiff", path.read_bytes())
    with pytest.raises(UnmixingError, match="CYX"):
        unmix.create_job(upload["upload_id"], DYES, MATRIX)


def test_reject_wrong_channel_count(service, unmix, tmp_path):
    upload = _ingest(service, tmp_path, channels=3)
    with pytest.raises(UnmixingError, match="通道"):
        unmix.create_job(upload["upload_id"], DYES, MATRIX)


def test_reject_non_uint16(service, unmix, tmp_path):
    upload = _ingest(service, tmp_path, dtype="uint8")
    with pytest.raises(UnmixingError, match="uint16"):
        unmix.create_job(upload["upload_id"], DYES, MATRIX)


def _ingest_large(service, sample_id: str, filename: str, blob: bytes) -> dict:
    """大文件走多分片入湖。"""
    upload = service.create_upload(sample_id, filename, len(blob), None)
    for number, chunk in enumerate(split_parts(blob), start=1):
        service.put_part(upload["upload_id"], number, chunk)
    service.complete(upload["upload_id"])
    return service.get_upload(upload["upload_id"])


def test_reject_decoded_buffer_over_16mib(service, unmix, tmp_path):
    # 2 通道 * 2048 * 2048 * 2 字节 = 16 MiB 恰好允许；再大即拒绝
    ok_path = tmp_path / "exact.ome.tiff"
    make_cyx_ome_tiff(ok_path, width=2048, height=2048)
    ok_upload = _ingest_large(
        service, "OK", "exact.ome.tiff", ok_path.read_bytes())
    record = unmix.create_job(ok_upload["upload_id"], DYES, MATRIX)
    assert record["status"] == "completed"

    big_path = tmp_path / "big.ome.tiff"
    make_cyx_ome_tiff(big_path, width=2049, height=2048)
    big_upload = _ingest_large(
        service, "BIG", "big.ome.tiff", big_path.read_bytes())
    with pytest.raises(UnmixingError, match="16 MiB"):
        unmix.create_job(big_upload["upload_id"], DYES, MATRIX)


def test_reject_all_saturated(service, unmix, tmp_path):
    data = np.full((2, 4, 4), 65535, dtype=np.uint16)
    path = tmp_path / "allsat.ome.tiff"
    with tifffile.TiffWriter(path) as w:
        w.write(data, photometric="minisblack", metadata={"axes": "CYX"})
    upload = complete_upload_with_blob(
        service, service.store, "SAT", "allsat.ome.tiff", path.read_bytes())
    with pytest.raises(UnmixingError, match="无有效像素"):
        unmix.create_job(upload["upload_id"], DYES, MATRIX)


def test_missing_ome_rejected_for_unmix(service, unmix, tmp_path, fake_s3):
    plain = tmp_path / "no-ome.tiff"
    tifffile.imwrite(plain, np.zeros((2, 8, 8), dtype=np.uint16))
    blob = plain.read_bytes()
    # 入湖校验只要求存在 OME 元数据，这里直接构造湖中对象 + completed 记录
    upload = service.create_upload("NOME", "no-ome.tiff", len(blob), None)
    service.put_part(upload["upload_id"], 1, blob)
    # 正常 complete 会拒绝；手动置 completed 并放对象，模拟历史数据
    service.store.put_object(upload["object_key"], blob)
    from app import db as db_mod
    with db_mod.connect() as conn:
        conn.execute(
            "UPDATE uploads SET status='completed', width=8, height=8, pages=2, "
            "dtype='uint16', axes='CYX' WHERE upload_id=?",
            (upload["upload_id"],))
    with pytest.raises(UnmixingError, match="OME"):
        unmix.create_job(upload["upload_id"], DYES, MATRIX)


# --------------------------------------------------------------- 原件/状态
def test_only_completed_upload_can_unmix(service, unmix, tmp_path):
    blob = make_cyx_ome_tiff(tmp_path / "open.ome.tiff")
    upload = service.create_upload("OPEN", "open.ome.tiff", len(blob), None)
    with pytest.raises(ConflictError, match="completed"):
        unmix.create_job(upload["upload_id"], DYES, MATRIX)


def test_unknown_upload_404(service, unmix):
    with pytest.raises(NotFoundError):
        unmix.create_job("ghost-id", DYES, MATRIX)


def test_source_read_failure_reports_storage_error(service, unmix, tmp_path):
    upload = _ingest(service, tmp_path)
    # 模拟原件在湖中丢失：读取失败必须明确报错且不留结果
    service.store.delete_object(upload["object_key"])
    with pytest.raises(Exception):
        unmix.create_job(upload["upload_id"], DYES, MATRIX)
    jobs = unmix.list_jobs(upload["upload_id"])
    assert jobs and jobs[0]["status"] == "failed"
    assert jobs[0]["result_object_key"] is not None
    # 失败任务不允许下载结果
    with pytest.raises(ConflictError):
        unmix.open_result_stream(jobs[0]["job_id"])


def test_failed_job_leaves_no_result_object(service, unmix, tmp_path, fake_s3):
    upload = _ingest(service, tmp_path)
    objects_before = dict(fake_s3.objects)
    with pytest.raises(UnmixingError):
        unmix.create_job(upload["upload_id"], DYES,
                         [[1.0, 1.0], [0.0, 0.001]])  # 条件数超限
    # 矩阵非法在任何对象写入前失败：湖中对象集合不变
    assert fake_s3.objects == objects_before


def test_list_and_get_job(service, unmix, tmp_path):
    upload = _ingest(service, tmp_path)
    j1 = unmix.create_job(upload["upload_id"], DYES, MATRIX)
    j2 = unmix.create_job(upload["upload_id"], ["GFP", "RFP"], MATRIX)
    jobs = unmix.list_jobs(upload["upload_id"])
    assert [j["job_id"] for j in jobs] == [j1["job_id"], j2["job_id"]]
    assert unmix.get_job(j1["job_id"])["dye_names"] == DYES
    with pytest.raises(NotFoundError):
        unmix.get_job("nope")
