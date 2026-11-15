"""双染料探测器串色解混（光谱解混合）核心。

输入约束（任一不满足即明确报错，不产出残缺结果）：

- 单序列（single series）、轴序恰好为 ``CYX``、恰好两个通道的 OME-TIFF；
- 像素类型必须为 ``uint16``；
- 解码后的像素缓冲不超过 16 MiB（配置项 ``MAX_DECODED_BYTES``）。

解混模型：两路探测器读数 ``y``（2 维）由两种染料的真实强度 ``x``
经 2×2 响应矩阵 ``A`` 线性混合得到 ``y = A·x``，其中 ``A[i, j]`` 表示
第 j 种染料对第 i 路探测器的响应。逐像素求解**非负最小二乘**（NNLS）：

    minimize ‖A·x − y‖²  subject to x ≥ 0

两路变量的 NNLS 只有 4 个候选活动集（零解、两个单染料解、双染料解），
全部按像素批量向量化计算后取残差最小且可行者，无需引入 scipy。

饱和规则：任一路原始读数达到 65535 的像素视为探测器饱和，两路解混结果
均记为 NaN；若所有像素都无效（全部饱和），抛出错误，不返回结果文件。

输出：两通道 float32 OME-TIFF，轴序 ``CYX``，OME-XML 通道名使用染料名称。
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import tifffile

from .config import (
    MAX_DECODED_BYTES,
    MAX_MATRIX_CONDITION,
    SATURATION_LEVEL,
    UNMIX_CHANNELS,
)
from .errors import UnmixingError

# 数值容差：小于该阈值的负值视为浮点误差钳为 0
_EPS = 1e-8


def validate_response_matrix(matrix: list[list[float]]) -> np.ndarray:
    """校验 2×2 响应矩阵：形状、有限性与条件数。

    矩阵布局：行=探测器（与图像通道顺序一致），列=染料（与染料名称顺序一致）。
    """
    try:
        arr = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise UnmixingError(
            f"响应矩阵必须是 {UNMIX_CHANNELS}×{UNMIX_CHANNELS} 数值数组：{exc}"
        ) from exc
    if arr.shape != (UNMIX_CHANNELS, UNMIX_CHANNELS):
        raise UnmixingError(
            f"响应矩阵必须是 {UNMIX_CHANNELS}×{UNMIX_CHANNELS}，实际形状 {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise UnmixingError("响应矩阵含非有限数（NaN/Inf），无法解混")

    # 条件数：奇异矩阵 -> inf，必然超过阈值被拦截
    condition = float(np.linalg.cond(arr))
    if not np.isfinite(condition) or condition > MAX_MATRIX_CONDITION:
        raise UnmixingError(
            f"响应矩阵条件数 {condition:g} 超过上限 {MAX_MATRIX_CONDITION:g}，"
            "两路染料响应近似共线，解混不稳定"
        )
    return arr


def load_two_channel_cyx_uint16(path: Path) -> np.ndarray:
    """读取并严格校验待解混图像，返回 CYX uint16 数组。

    仅在全部结构约束通过后才解码像素；解码量受 16 MiB 上限保护。
    """
    try:
        tif = tifffile.TiffFile(str(path))
    except Exception as exc:  # noqa: BLE001 - tifffile 异常类型较多
        raise UnmixingError(f"原件无法作为 TIFF 打开：{exc}") from exc

    try:
        if not tif.pages:
            raise UnmixingError("TIFF 不含任何页目录（IFD）")
        if getattr(tif, "ome_metadata", None) is None and not _has_ome_description(tif):
            raise UnmixingError("缺少 OME-XML 元数据，不是 OME-TIFF")
        if len(tif.series) != 1:
            raise UnmixingError(
                f"只支持单序列 OME-TIFF，实际含 {len(tif.series)} 个序列"
            )

        series = tif.series[0]
        axes = "".join(series.axes) if series.axes else "".join(series.get_axes())
        # 轴序与维度一并强制：单序列 CYX 必须恰好三维，杜绝 TCYX/ZCYX 等混入
        if series.ndim != 3 or axes != "CYX":
            raise UnmixingError(
                f"只支持轴序恰好为 CYX 的三维单序列图像，"
                f"实际轴序为 {axes!r}、维度为 {series.ndim}"
            )
        if series.shape[0] != UNMIX_CHANNELS:
            raise UnmixingError(
                f"只支持恰好 {UNMIX_CHANNELS} 个通道的图像，实际形状 {series.shape}"
            )
        if str(series.dtype) != "uint16":
            raise UnmixingError(f"只支持 uint16 像素，实际 dtype 为 {series.dtype}")

        channels, height, width = (int(v) for v in series.shape)
        decoded_bytes = channels * height * width * np.dtype(np.uint16).itemsize
        if decoded_bytes > MAX_DECODED_BYTES:
            raise UnmixingError(
                f"解码后像素缓冲 {decoded_bytes} 字节超过上限 "
                f"{MAX_DECODED_BYTES} 字节（16 MiB）"
            )

        try:
            data = series.asarray()
        except Exception as exc:  # noqa: BLE001 - 像素解码失败属于坏原件
            raise UnmixingError(f"像素解码失败，原件可能已损坏：{exc}") from exc
    finally:
        tif.close()

    data = np.asarray(data)
    if data.dtype != np.uint16 or data.ndim != 3 or data.shape[0] != UNMIX_CHANNELS:
        # 防御性兜底：正常情况下前面的约束已拦截
        raise UnmixingError("解码结果不符合 CYX/两通道/uint16 约束")
    return data


def _has_ome_description(tif: tifffile.TiffFile) -> bool:
    for page in tif.pages:
        desc = page.description or ""
        head = desc.lstrip()[:200]
        if "OME:" in head or "<OME" in head or "<ome:" in head:
            return True
    return False


def nnls_two_dyes(y: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """逐像素求解两路染料的非负最小二乘。

    ``y`` 形状 (2, N)，列为各像素的两路探测器读数；
    返回 ``x`` 形状 (2, N)，行为两种染料的非负强度（float64）。

    两个变量的 NNLS 可行活动集只有 4 个，全部批量计算后选残差最小者。
    """
    n_pixels = y.shape[1]
    a11, a12 = matrix[0, 0], matrix[0, 1]
    a21, a22 = matrix[1, 0], matrix[1, 1]
    y1, y2 = y[0], y[1]

    # 列向量归一 AᵀA 与 Aᵀy（对每个像素相同的矩阵部分 + 像素相关部分）
    ata00 = a11 * a11 + a21 * a21
    ata11 = a12 * a12 + a22 * a22
    ata01 = a11 * a12 + a21 * a22
    aty0 = a11 * y1 + a21 * y2
    aty1 = a12 * y1 + a22 * y2
    det = ata00 * ata11 - ata01 * ata01  # 条件数校验已保证 det > 0

    best = np.zeros((UNMIX_CHANNELS, n_pixels), dtype=np.float64)
    # 零解残差 ‖y‖²，作为初始最优
    best_res = y1 * y1 + y2 * y2

    def consider(candidate: np.ndarray, residual: np.ndarray) -> None:
        # 两分量均非负（容差内）才是可行解；严格更优才替换，并列时保留先入者
        feasible = (candidate[0] >= -_EPS) & (candidate[1] >= -_EPS)
        better = residual < best_res - _EPS
        take = feasible & better
        best[:, take] = candidate[:, take]
        best_res[take] = residual[take]

    # 候选 1：仅染料 0（x1=0，x0 = Aᵀy[0] / AᵀA[0,0]）
    x0 = aty0 / ata00
    cand = np.vstack([x0, np.zeros_like(x0)])
    r1 = a11 * x0 - y1
    r2 = a21 * x0 - y2
    consider(cand, r1 * r1 + r2 * r2)

    # 候选 2：仅染料 1（x0=0，x1 = Aᵀy[1] / AᵀA[1,1]）
    x1 = aty1 / ata11
    cand = np.vstack([np.zeros_like(x1), x1])
    r1 = a12 * x1 - y1
    r2 = a22 * x1 - y2
    consider(cand, r1 * r1 + r2 * r2)

    # 候选 3：双染料（无约束最小二乘解，可行时即 NNLS 解）
    x0_full = (ata11 * aty0 - ata01 * aty1) / det
    x1_full = (ata00 * aty1 - ata01 * aty0) / det
    cand = np.vstack([x0_full, x1_full])
    r1 = a11 * x0_full + a12 * x1_full - y1
    r2 = a21 * x0_full + a22 * x1_full - y2
    consider(cand, r1 * r1 + r2 * r2)

    # 钳掉容差内的微小负值，保证输出严格非负（NaN 像素在调用方处理）
    np.maximum(best, 0.0, out=best)
    return best


def unmix_image(data: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """对 CYX uint16 图像执行解混，返回 CYX float32 结果（含 NaN 标记）。"""
    channels, height, width = data.shape
    # 任一路原始读数达到饱和电平 -> 该像素两路结果均为 NaN
    saturated = np.any(data >= SATURATION_LEVEL, axis=0)
    valid = ~saturated
    if not np.any(valid):
        raise UnmixingError(
            "全部像素的某一路探测器读数达到 65535（饱和），无有效像素可解混"
        )

    # 输出预填 NaN：饱和像素天然保持 NaN，仅对有效像素求解
    result = np.full((channels, height, width), np.nan, dtype=np.float32)

    y = data[:, valid].astype(np.float64, copy=False)  # (2, N_valid)
    x = nnls_two_dyes(y, matrix).astype(np.float32)    # (2, N_valid)
    result[:, valid] = x
    return result


def write_float32_ome_tiff(result: np.ndarray, dye_names: list[str]) -> bytes:
    """把 CYX float32 解混结果编码为带染料通道名的 OME-TIFF 字节串。"""
    if result.dtype != np.float32 or result.ndim != 3:
        raise UnmixingError("解混结果必须是 CYX float32 数组")
    if len(dye_names) != result.shape[0]:
        raise UnmixingError("染料名称数量必须与输出通道数一致")

    buffer = io.BytesIO()
    try:
        with tifffile.TiffWriter(buffer, ome=True) as writer:
            writer.write(
                result,
                photometric="minisblack",
                metadata={
                    "axes": "CYX",
                    "Name": "unmixed",
                    "Channel": {"Name": list(dye_names)},
                },
            )
    except Exception as exc:  # noqa: BLE001 - 编码失败明确报错，不返回残件
        raise UnmixingError(f"解混结果 OME-TIFF 编码失败：{exc}") from exc
    return buffer.getvalue()
