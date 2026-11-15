#!/usr/bin/env python3
"""端到端故障演示客户端（仅使用标准库，容器内可直接运行）。

演示清单：
  1. 小图完整上传 -> 合并核验 -> 尺寸/原件开放
  2. 大图分片中断 -> 查缺片 -> 续传完成，不重传整图
  3. 同片重传幂等；异内容返回 409
  4. 放弃清除远端分片，迟到写入被拒（无残件）
  5. 进程在“分片已写 S3、落库前”崩溃 -> 重启对账 -> 仅补缺片
  6. 进程在“合并成功、回执前”崩溃 -> 重启续完 -> 重试幂等不重复发布
  7. 坏 OME-TIFF 被拒，远端对象删除，原件不开放
  8. 双通道 CYX uint16 双染料图像 NNLS 解混，饱和像素 NaN，结果可下载
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = os.environ.get("API_BASE", "http://127.0.0.1:8000")
SAMPLE_DIR = Path(os.environ.get("SAMPLE_DIR", "/sample-data"))
PART_SIZE = 8 * 1024 * 1024


# --------------------------------------------------------------------- HTTP 工具
def _request(method: str, path: str, data: bytes | None = None,
             headers: dict | None = None):
    req = urllib.request.Request(
        API + path, data=data, method=method, headers=headers or {}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read()
            return resp.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"raw": body[:200].decode("utf-8", "replace")}
        return exc.code, payload


def step(title: str) -> None:
    print(f"\n=== {title} ===")


def check(cond: bool, message: str) -> None:
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {message}")
    if not cond:
        raise SystemExit(f"演示断言失败: {message}")


def wait_api_ready(timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(API + "/health", timeout=3) as resp:
                if resp.status == 200:
                    return
        except OSError:
            time.sleep(1)
    raise SystemExit("API 在等待时间内未就绪")


def split_parts(blob: bytes) -> list[bytes]:
    return [blob[i:i + PART_SIZE] for i in range(0, len(blob), PART_SIZE)]


# --------------------------------------------------------------------- API 封装
def create_upload(sample_id: str, filename: str, total: int,
                  sha256: str | None = None) -> dict:
    payload = {"sample_id": sample_id, "filename": filename,
               "total_bytes": total}
    if sha256:
        payload["content_sha256"] = sha256
    status, body = _request("POST", "/uploads",
                            json.dumps(payload).encode(),
                            {"Content-Type": "application/json"})
    check(status == 201, f"创建上传 -> {status}")
    return body


def put_part(uid: str, number: int, chunk: bytes):
    return _request("PUT", f"/uploads/{uid}/parts/{number}", chunk,
                    {"Content-Type": "application/octet-stream"})


def get_parts(uid: str) -> dict:
    _, body = _request("GET", f"/uploads/{uid}/parts")
    return body


def complete(uid: str):
    return _request("POST", f"/uploads/{uid}/complete")


def abort(uid: str):
    return _request("POST", f"/uploads/{uid}/abort")


def arm_fault(fault: str, part_number: int | None = None):
    payload = {"fault": fault}
    if part_number is not None:
        payload["part_number"] = part_number
    return _request("POST", "/dev/faults", json.dumps(payload).encode(),
                    {"Content-Type": "application/json"})


def s3_inflight_uploads() -> list[dict]:
    """通过 boto3 查询 MinIO 上仍在进行的 multipart 上传（残件检查）。"""
    import boto3
    from botocore.config import Config
    client = boto3.client(
        "s3", endpoint_url=os.environ.get("S3_ENDPOINT", "http://minio:9000"),
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=os.environ.get("S3_SECRET_KEY", "minioadmin"),
        config=Config(signature_version="s3v4",
                      s3={"addressing_style": "path"}),
    )
    resp = client.list_multipart_uploads(Bucket="ome-lake")
    return resp.get("Uploads") or []


def s3_head(key: str) -> dict:
    import boto3
    from botocore.config import Config
    client = boto3.client(
        "s3", endpoint_url=os.environ.get("S3_ENDPOINT", "http://minio:9000"),
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=os.environ.get("S3_SECRET_KEY", "minioadmin"),
        config=Config(signature_version="s3v4",
                      s3={"addressing_style": "path"}),
    )
    try:
        return client.head_object(Bucket="ome-lake", Key=key)
    except Exception:
        return {}


# --------------------------------------------------------------------- 场景
def scenario_1_happy_path():
    step("场景 1：小图完整上传，合并核验后开放尺寸与原件")
    blob = (SAMPLE_DIR / "good-small.ome.tiff").read_bytes()
    up = create_upload("DEMO-1", "good-small.ome.tiff", len(blob))
    status, part = put_part(up["upload_id"], 1, blob)
    check(status == 200 and part["duplicate"] is False, "唯一分片被接纳")
    status, done = complete(up["upload_id"])
    check(status == 200 and done["status"] == "completed", "合并核验完成")
    print(f"    提取尺寸: {done['width']}x{done['height']} "
          f"pages={done['pages']} dtype={done['dtype']} axes={done['axes']}")
    status, dims = _request("GET", f"/uploads/{up['upload_id']}/dimensions")
    check(status == 200 and dims["width"] == done["width"], "尺寸查询已开放")
    status, _, headers = _request_with_headers(
        "GET", f"/uploads/{up['upload_id']}/object")
    check(status == 200, "原件下载已开放")
    check(int(headers["Content-Length"]) == len(blob), "原件字节数与声明一致")


def _request_with_headers(method: str, path: str):
    req = urllib.request.Request(API + path, method=method)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.status, resp.read(), resp.headers


def scenario_2_resume():
    step("场景 2：大图分片中断 -> 查缺片 -> 续传（不重传整图）")
    blob = (SAMPLE_DIR / "good-large.ome.tiff").read_bytes()
    parts = split_parts(blob)
    up = create_upload("DEMO-2", "good-large.ome.tiff", len(blob))
    check(up["expected_parts"] == len(parts),
          f"规划 {len(parts)} 个分片（每片 8 MiB，末片 {len(parts[-1])} 字节）")
    # “网络中断”：只传前 2 片
    for i, chunk in enumerate(parts[:2], start=1):
        status, _ = put_part(up["upload_id"], i, chunk)
        check(status == 200, f"分片 {i} 接纳")
    view = get_parts(up["upload_id"])
    print(f"    中断后缺片: {view['missing_parts']}")
    check(view["missing_parts"] == list(range(3, len(parts) + 1)),
          "缺片清单正确")
    # 续传剩余片
    for i, chunk in enumerate(parts[2:], start=3):
        status, _ = put_part(up["upload_id"], i, chunk)
        check(status == 200, f"续传分片 {i} 接纳")
    status, done = complete(up["upload_id"])
    check(status == 200 and done["status"] == "completed", "续传后合并成功")
    _, downloaded, _ = _request_with_headers(
        "GET", f"/uploads/{up['upload_id']}/object")
    check(downloaded == blob, "湖内原件与本地文件逐字节一致")


def scenario_3_idempotency_conflict():
    step("场景 3：同片同字节幂等；同片异内容 409")
    blob = (SAMPLE_DIR / "good-small.ome.tiff").read_bytes()
    up = create_upload("DEMO-3", "good-small.ome.tiff", len(blob))
    s1, p1 = put_part(up["upload_id"], 1, blob)
    s2, p2 = put_part(up["upload_id"], 1, blob)
    check(s2 == 200 and p2["duplicate"] is True and
          p2["sha256"] == p1["sha256"], "同片重传幂等返回 200")
    altered = bytearray(blob)
    altered[0] ^= 0xFF
    s3, err = put_part(up["upload_id"], 1, bytes(altered))
    check(s3 == 409 and err["error"]["code"] == "part_conflict",
          "异内容重传返回 409 且不覆盖")
    status, done = complete(up["upload_id"])
    check(status == 200, "原内容未被污染，仍可正常完成")


def scenario_4_abort():
    step("场景 4：放弃清除远端分片，迟到写入被拒，无残件")
    blob = (SAMPLE_DIR / "good-small.ome.tiff").read_bytes()
    up = create_upload("DEMO-4", "good-small.ome.tiff", len(blob))
    put_part(up["upload_id"], 1, blob)
    status, _ = abort(up["upload_id"])
    check(status == 200, "放弃成功")
    time.sleep(1)
    inflight = s3_inflight_uploads()
    check(not any(u["Key"] == up["object_key"] for u in inflight),
          "远端 multipart 分片已清除")
    status, err = put_part(up["upload_id"], 1, blob)
    check(status == 410, f"迟到分片被拒（HTTP {status}）")
    status, _ = _request("GET", f"/uploads/{up['upload_id']}/dimensions")
    check(status == 409, "放弃任务不开放任何原件查询")


def scenario_5_crash_after_part():
    step("场景 5：分片写 S3 后、落库前崩溃 -> 重启 -> 仅补缺片")
    blob = (SAMPLE_DIR / "good-large.ome.tiff").read_bytes()
    parts = split_parts(blob)
    up = create_upload("DEMO-5", "good-large.ome.tiff", len(blob))
    uid = up["upload_id"]
    for i, chunk in enumerate(parts[:2], start=1):
        put_part(uid, i, chunk)
    # 第 3 片写入后立即 SIGTERM（远端已写、本地无记录）
    status, armed = arm_fault("crash_after_part", 3)
    check(status == 200, f"故障注入已布防: {armed}")
    try:
        put_part(uid, 3, parts[2])
    except Exception:
        pass
    print("    API 进程已被 SIGTERM，等待容器重启与启动恢复……")
    wait_api_ready()
    time.sleep(2)  # 等待 lifespan 恢复流程完成
    view = get_parts(uid)
    print(f"    重启后缺片: {view['missing_parts']}")
    check(3 in view["missing_parts"], "崩溃片被正确识别为缺片")
    # 只补缺失片，前 2 片无需重传
    for i in view["missing_parts"]:
        status, _ = put_part(uid, i, parts[i - 1])
        check(status == 200, f"补缺片 {i} 成功")
    status, done = complete(uid)
    check(status == 200 and done["status"] == "completed", "恢复后合并成功")
    _, downloaded, _ = _request_with_headers("GET", f"/uploads/{uid}/object")
    check(downloaded == blob, "恢复后原件完整一致")


def scenario_6_crash_after_merge():
    step("场景 6：S3 合并成功后、回执前崩溃 -> 重启续完，重试不重复发布")
    blob = (SAMPLE_DIR / "good-small.ome.tiff").read_bytes()
    up = create_upload("DEMO-6", "good-small.ome.tiff", len(blob))
    uid = up["upload_id"]
    put_part(uid, 1, blob)
    status, _ = arm_fault("crash_after_merge")
    check(status == 200, "合并后崩溃故障已布防")
    try:
        complete(uid)
    except Exception:
        pass
    print("    合并已成功但回执前进程终止，等待重启恢复……")
    wait_api_ready()
    time.sleep(2)
    status, body = _request("GET", f"/uploads/{uid}")
    check(body["status"] == "completed",
          f"启动恢复已续完任务（状态={body['status']}）")
    # 客户端重试 complete：幂等，不重新合并/发布
    status, done = complete(uid)
    check(status == 200 and done["status"] == "completed",
          "重试 complete 幂等成功")
    head = s3_head(up["object_key"])
    check(head and head["ContentLength"] == len(blob),
          "湖中仅一个对象，无重复发布")


def scenario_7_bad_image():
    step("场景 7：缺 OME 元数据的坏图被拒，对象删除，原件不开放")
    blob = (SAMPLE_DIR / "bad-no-ome.tiff").read_bytes()
    up = create_upload("DEMO-7", "bad-no-ome.tiff", len(blob))
    put_part(up["upload_id"], 1, blob)
    status, err = complete(up["upload_id"])
    check(status == 422, f"坏图返回 422: {err['error']['message'][:48]}...")
    status, info = _request("GET", f"/uploads/{up['upload_id']}")
    check(info["status"] == "invalid", "任务标记为 invalid")
    check(s3_head(up["object_key"]) == {}, "远端坏图对象已删除，未入湖")
    status, _ = _request("GET", f"/uploads/{up['upload_id']}/object")
    check(status == 409, "原件查询不开放")


def scenario_8_unmixing():
    step("场景 8：双染料 CYX uint16 图像 NNLS 解混，饱和像素 NaN，结果可下载")
    blob = (SAMPLE_DIR / "two-dye-cyx.ome.tiff").read_bytes()
    up = create_upload("DEMO-8", "two-dye-cyx.ome.tiff", len(blob))
    put_part(up["upload_id"], 1, blob)
    status, done = complete(up["upload_id"])
    check(status == 200 and done["status"] == "completed", "原件入湖完成")
    uid = up["upload_id"]

    matrix = [[1.0, 0.25], [0.4, 1.0]]
    payload = json.dumps({"dye_names": ["Dye-A", "Dye-B"],
                          "response_matrix": matrix}).encode()
    status, job = _request("POST", f"/uploads/{uid}/unmix", payload,
                           {"Content-Type": "application/json"})
    check(status == 201 and job["status"] == "completed",
          f"解混任务完成（HTTP {status}）")
    print(f"    有效像素 {job['valid_pixels']}，饱和像素 {job['saturated_pixels']}")
    check(job["saturated_pixels"] == 2, "两个饱和像素被识别")
    check(job["width"] == 128 and job["height"] == 128, "尺寸回传正确")

    # 下载结果并校验：float32、两通道、OME 通道名为染料名
    status, result_blob, headers = _request_with_headers(
        "GET", f"/unmix-jobs/{job['job_id']}/result")
    check(status == 200, "解混结果可下载")
    check(headers["Content-Type"] == "image/tiff", "Content-Type 为 image/tiff")
    check(int(headers["Content-Length"]) == job["result_bytes"],
          "结果字节数与记录一致")
    _assert_unmixed_tiff(result_blob)

    # 任务可查询、可列举
    status, got = _request("GET", f"/unmix-jobs/{job['job_id']}")
    check(status == 200 and got["dye_names"] == ["Dye-A", "Dye-B"], "任务查询")
    status, listed = _request("GET", f"/uploads/{uid}/unmix")
    check(status == 200 and len(listed) == 1, "按上传列举解混任务")

    # 原件保持不变
    _, original, _ = _request_with_headers("GET", f"/uploads/{uid}/object")
    check(original == blob, "原件逐字节未被解混改动")

    # 非法矩阵（条件数 2000 > 1000）明确 422，且不产生结果对象
    bad = json.dumps({"dye_names": ["Dye-A", "Dye-B"],
                      "response_matrix": [[1.0, 1.0], [0.0, 0.001]]}).encode()
    status, err = _request("POST", f"/uploads/{uid}/unmix", bad,
                           {"Content-Type": "application/json"})
    check(status == 422 and "条件数" in err["error"]["message"],
          f"病态矩阵被 422 拒绝: {err['error']['message'][:40]}...")


def _assert_unmixed_tiff(blob: bytes) -> None:
    """仅用 numpy/tifffile 校验下载到的解混产物（容器内已安装）。"""
    import io

    import tifffile

    with tifffile.TiffFile(io.BytesIO(blob)) as tif:
        arr = tif.asarray()
        check(str(arr.dtype) == "float32", "输出 dtype 为 float32")
        check(arr.shape == (2, 128, 128), f"输出形状为 2×128×128，实际 {arr.shape}")
        check("".join(tif.series[0].axes) == "CYX", "输出轴序为 CYX")
        ome = tif.ome_metadata
        check('Name="Dye-A"' in ome and 'Name="Dye-B"' in ome,
              "OME-XML 通道名使用染料名称")
        saturated_nan = int(__import__("numpy").isnan(arr[0]).sum())
        check(saturated_nan == 2, f"饱和像素两路均为 NaN（NaN 数={saturated_nan}）")


def main() -> None:
    wait_api_ready()
    print(f"API: {API}  样例目录: {SAMPLE_DIR}")
    scenario_1_happy_path()
    scenario_2_resume()
    scenario_3_idempotency_conflict()
    scenario_4_abort()
    scenario_5_crash_after_part()
    scenario_6_crash_after_merge()
    scenario_7_bad_image()
    scenario_8_unmixing()
    print("\n全部演示场景通过。可在 http://localhost:8000/docs 手动操作全流程。")


if __name__ == "__main__":
    main()
