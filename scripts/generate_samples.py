#!/usr/bin/env python3
"""生成演示用合成 OME-TIFF（纯算法数据，无任何隐私图像）。

产物：
  good-large.ome.tiff    多页大文件（约 24 MiB，演示 8 MiB 分片续传）
  good-small.ome.tiff    单页小文件（末片短分片）
  bad-no-ome.tiff        普通 TIFF，缺 OME 元数据（演示坏图拦截）
  two-dye-cyx.ome.tiff   双通道 CYX uint16，已知 2×2 矩阵混合（演示解混），
                         含两个饱和像素（65535）用于演示 NaN 规则
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import tifffile

OUT_DIR = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/sample-data")

# 两染料对两探测器的响应矩阵：行=探测器，列=染料
DEMO_MATRIX = np.array([[1.0, 0.25], [0.4, 1.0]], dtype=np.float64)


def _two_dye_cyx() -> np.ndarray:
    """按已知矩阵合成双染料双通道读数，确定性梯度 + 饱和点。"""
    h = w = 128
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dye0 = (400.0 + 6.0 * xx + 2.0 * yy).clip(0, 6000)
    dye1 = (300.0 + 3.0 * xx + 5.0 * yy).clip(0, 6000)
    true_intensities = np.stack([dye0, dye1], axis=0)       # (2, H, W)
    mixed = np.einsum("ij,jyx->iyx", DEMO_MATRIX, true_intensities)
    data = np.clip(np.rint(mixed), 0, 65534).astype(np.uint16)
    # 两个饱和像素：解混后两路都应为 NaN
    data[0, 0, 0] = 65535
    data[1, 10, 10] = 65535
    return data


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # 3 页 2048x2048 uint16：确定性梯度 + 噪声，约 24 MiB，4 个分片
    large = (
        np.arange(3 * 2048 * 2048, dtype=np.uint16).reshape(3, 2048, 2048)
        + rng.integers(0, 64, size=(3, 2048, 2048), dtype=np.uint16)
    )
    with tifffile.TiffWriter(OUT_DIR / "good-large.ome.tiff") as w:
        w.write(large, photometric="minisblack", metadata={"axes": "ZYX"})

    # 小文件：单页 256x256 uint8，仅 64 KiB，单分片（末片可短）
    small = rng.integers(0, 256, size=(256, 256), dtype=np.uint8)
    with tifffile.TiffWriter(OUT_DIR / "good-small.ome.tiff") as w:
        w.write(small, photometric="minisblack", metadata={"axes": "YX"})

    # 坏图：结构合法的 TIFF，但不含 OME-XML
    tifffile.imwrite(
        OUT_DIR / "bad-no-ome.tiff",
        np.zeros((2, 64, 64), dtype=np.uint8),
        photometric="minisblack",
    )

    # 双染料双通道 CYX uint16（解混演示专用）
    with tifffile.TiffWriter(OUT_DIR / "two-dye-cyx.ome.tiff") as w:
        w.write(_two_dye_cyx(), photometric="minisblack",
                metadata={"axes": "CYX", "Name": "two-dye-sample"})

    for path in sorted(OUT_DIR.glob("*.tiff")):
        print(f"generated {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
