"""解混 API 的 HTTP 层测试：请求/响应语义、状态码与下载。"""
from __future__ import annotations

import io

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient

from app import main as main_mod

from .conftest import make_cyx_ome_tiff

MATRIX = [[1.0, 0.25], [0.4, 1.0]]
DYES = ["DAPI", "Cy3"]


@pytest.fixture
def client(fake_s3, monkeypatch):
    object.__setattr__(main_mod.settings, "recover_on_startup", False)
    monkeypatch.setattr(main_mod, "S3Store", lambda: fake_s3)
    with TestClient(main_mod.app) as c:
        yield c


@pytest.fixture
def ingested(client, tmp_path):
    blob = make_cyx_ome_tiff(tmp_path / "cyx.ome.tiff", width=36, height=24,
                             saturated=True)
    resp = client.post("/uploads", json={
        "sample_id": "UMIX", "filename": "cyx.ome.tiff",
        "total_bytes": len(blob)})
    assert resp.status_code == 201
    upload = resp.json()
    assert client.put(
        f"/uploads/{upload['upload_id']}/parts/1", content=blob).status_code == 200
    assert client.post(
        f"/uploads/{upload['upload_id']}/complete").status_code == 200
    return upload, blob


def test_unmix_full_http_flow(client, ingested):
    upload, original = ingested
    uid = upload["upload_id"]

    r = client.post(f"/uploads/{uid}/unmix",
                    json={"dye_names": DYES, "response_matrix": MATRIX})
    assert r.status_code == 201, r.text
    job = r.json()
    assert job["status"] == "completed"
    assert job["dye_names"] == DYES
    assert job["saturated_pixels"] == 2
    assert job["valid_pixels"] == 36 * 24 - 2

    # 作业可查询
    got = client.get(f"/unmix-jobs/{job['job_id']}")
    assert got.status_code == 200 and got.json()["job_id"] == job["job_id"]
    listed = client.get(f"/uploads/{uid}/unmix")
    assert listed.status_code == 200 and len(listed.json()) == 1

    # 下载产物：float32 两通道 OME-TIFF，通道名为染料名，NaN 保留
    d = client.get(f"/unmix-jobs/{job['job_id']}/result")
    assert d.status_code == 200
    assert d.headers["content-type"] == "image/tiff"
    assert "attachment" in d.headers["content-disposition"]
    assert int(d.headers["content-length"]) == job["result_bytes"]
    with tifffile.TiffFile(io.BytesIO(d.content)) as tif:
        result = tif.asarray()
        assert result.dtype == np.float32 and result.shape == (2, 24, 36)
        assert tif.series[0].axes == "CYX"
        assert 'Name="DAPI"' in tif.ome_metadata and 'Name="Cy3"' in tif.ome_metadata
        assert np.all(np.isnan(result[:, 0, 0]))

    # 原件仍可逐字节下载，未被解混改动
    obj = client.get(f"/uploads/{uid}/object")
    assert obj.status_code == 200 and obj.content == original


def test_unmix_bad_matrix_returns_422(client, ingested):
    uid = ingested[0]["upload_id"]
    r = client.post(f"/uploads/{uid}/unmix", json={
        "dye_names": DYES,
        "response_matrix": [[1.0, 1.0], [0.0, 0.001]]})  # 条件数 2000
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "unmixing_error"
    assert "条件数" in err["message"]


def test_unmix_non_finite_matrix_422(client, ingested):
    uid = ingested[0]["upload_id"]
    r = client.post(f"/uploads/{uid}/unmix", json={
        "dye_names": DYES,
        "response_matrix": [[1.0, None], [0.0, 1.0]]})
    # None 无法解析为 float -> 请求体 422；NaN 字符串可解析但被业务拦截
    assert r.status_code == 422

    r2 = client.post(f"/uploads/{uid}/unmix", json={
        "dye_names": DYES,
        "response_matrix": [[1.0, "NaN"], [0.0, 1.0]]})
    assert r2.status_code == 422
    assert "非有限" in r2.json()["error"]["message"]


def test_unmix_wrong_shape_image_422(client, tmp_path):
    # 三通道 ZYX 图像入湖成功，但解混必须 422
    blob = make_cyx_ome_tiff(tmp_path / "zyx.ome.tiff", axes="ZYX", channels=3)
    upload = client.post("/uploads", json={
        "sample_id": "BAD", "filename": "zyx.ome.tiff",
        "total_bytes": len(blob)}).json()
    client.put(f"/uploads/{upload['upload_id']}/parts/1", content=blob)
    client.post(f"/uploads/{upload['upload_id']}/complete")
    r = client.post(f"/uploads/{upload['upload_id']}/unmix",
                    json={"dye_names": DYES, "response_matrix": MATRIX})
    assert r.status_code == 422 and "CYX" in r.json()["error"]["message"]


def test_unmix_unknown_upload_404(client):
    r = client.post("/uploads/missing/unmix",
                    json={"dye_names": DYES, "response_matrix": MATRIX})
    assert r.status_code == 404


def test_unmix_request_validation(client, ingested):
    uid = ingested[0]["upload_id"]
    # 染料名称数量必须恰好 2 个
    r = client.post(f"/uploads/{uid}/unmix",
                    json={"dye_names": ["OnlyOne"], "response_matrix": MATRIX})
    assert r.status_code == 422
    # 空染料名被拒
    r2 = client.post(f"/uploads/{uid}/unmix", json={
        "dye_names": ["", "Cy3"], "response_matrix": MATRIX})
    assert r2.status_code == 422


def test_unmix_openapi_documents_routes(client):
    spec = client.get("/openapi.json").json()
    for path in ("/uploads/{upload_id}/unmix",
                 "/unmix-jobs/{job_id}",
                 "/unmix-jobs/{job_id}/result"):
        assert path in spec["paths"]
