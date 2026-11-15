"""解混业务逻辑：读取湖内原件 -> 严格校验 -> NNLS 解混 -> 结果回流入湖。

设计约束：

- 只接受 ``completed`` 状态上传的原件；原件始终只读，绝不修改或删除；
- 任何一步失败都明确抛错，不写入结果对象，不返回残缺产物；
- 结果作为独立对象写回同一数据湖，通过下载端点流式提供；
- 同一 upload_id 的解混与上传生命周期操作共用服务层每任务锁。
"""
from __future__ import annotations

import json
import uuid
from typing import Any

import numpy as np

from . import db
from .config import settings
from .errors import ConflictError, NotFoundError
from .service import _upload_lock
from .unmixing import (
    load_two_channel_cyx_uint16,
    unmix_image,
    validate_response_matrix,
    write_float32_ome_tiff,
)


class UnmixingService:
    def __init__(self, store: Any):
        self.store = store

    # ------------------------------------------------------------------ 入口
    def create_job(self, upload_id: str, dye_names: list[str],
                   matrix_raw: list[list[float]]) -> dict[str, Any]:
        # 矩阵先做纯数值校验：非法矩阵在触碰原件前即 422
        matrix = validate_response_matrix(matrix_raw)

        # 上传不存在/未完成：直接报错，不落任何任务记录
        upload = self._get_completed_upload(upload_id)

        job_id = uuid.uuid4().hex
        result_key = (
            f"unmixed/{upload['sample_id']}/{upload_id}/"
            f"{job_id}/unmixed.ome.tiff"
        )
        with db.connect() as conn:
            conn.execute(
                """INSERT INTO unmix_jobs (job_id, upload_id, status, dye_names,
                       response_matrix, result_object_key, error_message)
                   VALUES (?, ?, 'failed', ?, ?, ?, NULL)""",
                (job_id, upload_id, json.dumps(dye_names, ensure_ascii=False),
                 json.dumps(matrix_raw), result_key),
            )

        try:
            with _upload_lock(upload_id):
                record = self._run_job(job_id, upload, dye_names, matrix, result_key)
        except Exception as exc:
            # 处理阶段失败（原件读取/图像约束/全像素无效/编码/回写）：
            # 留痕为 failed 并抛出明确错误，绝不返回残缺结果
            self._mark_failed(job_id, str(exc))
            raise
        return record

    # ------------------------------------------------------------------ 执行
    def _run_job(self, job_id: str, upload: dict[str, Any],
                 dye_names: list[str], matrix: np.ndarray,
                 result_key: str) -> dict[str, Any]:
        settings.tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = settings.tmp_dir / f"unmix-{job_id}.ome.tiff"
        try:
            # 原件读取失败（对象缺失/存储故障）在此抛出，不产出任何结果
            self.store.download_to_file(upload["object_key"], str(tmp_path))

            data = load_two_channel_cyx_uint16(tmp_path)
            channels, height, width = data.shape
            total_pixels = height * width

            result = unmix_image(data, matrix)
            saturated = int(np.count_nonzero(np.any(data >= 65535, axis=0)))
            valid_pixels = total_pixels - saturated

            blob = write_float32_ome_tiff(result, dye_names)
            # 结果对象独立入湖；原件对象键不同，保持不变
            self.store.put_object(result_key, blob, content_type="image/tiff")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        with db.connect() as conn:
            conn.execute(
                """UPDATE unmix_jobs SET status='completed', result_object_key=?,
                       result_bytes=?, width=?, height=?, saturated_pixels=?,
                       valid_pixels=?, error_message=NULL WHERE job_id=?""",
                (result_key, len(blob), width, height, saturated,
                 valid_pixels, job_id),
            )
        return self.get_job(job_id)

    # ------------------------------------------------------------------ 查询
    def get_job(self, job_id: str) -> dict[str, Any]:
        with db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM unmix_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"解混任务不存在: {job_id}")
        return self._row_to_response(row)

    def list_jobs(self, upload_id: str) -> list[dict[str, Any]]:
        # 上传本身不存在应 404；存在但无任务返回空列表
        self._get_completed_upload(upload_id)
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM unmix_jobs WHERE upload_id=? ORDER BY created_at",
                (upload_id,),
            ).fetchall()
        return [self._row_to_response(r) for r in rows]

    def open_result_stream(self, job_id: str):
        record = self.get_job(job_id)
        if record["status"] != "completed":
            raise ConflictError(
                f"解混任务状态为 {record['status']}，无结果可下载："
                f"{record.get('error_message')}",
                code="unmix_result_unavailable",
            )
        body = self.store.get_object_stream(record["result_object_key"])
        return record, body

    # ------------------------------------------------------------------ 工具
    def _get_completed_upload(self, upload_id: str) -> dict[str, Any]:
        with db.connect() as conn:
            upload = db.get_upload(conn, upload_id)
        if upload is None:
            raise NotFoundError(f"上传任务不存在: {upload_id}")
        if upload["status"] != "completed":
            raise ConflictError(
                f"上传任务状态为 {upload['status']}，仅 completed 原件可提交解混",
                code="not_ready",
            )
        return upload

    @staticmethod
    def _mark_failed(job_id: str, message: str) -> None:
        with db.connect() as conn:
            conn.execute(
                "UPDATE unmix_jobs SET status='failed', error_message=? WHERE job_id=?",
                (message, job_id),
            )

    @staticmethod
    def _row_to_response(row: Any) -> dict[str, Any]:
        item = dict(row)
        item["dye_names"] = json.loads(item["dye_names"])
        item["response_matrix"] = json.loads(item["response_matrix"])
        item.pop("created_at", None)
        return item
