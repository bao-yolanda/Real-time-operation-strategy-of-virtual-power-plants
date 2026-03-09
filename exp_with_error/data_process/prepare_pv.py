"""
光伏出力数据准备
"""
import numpy as np
import scipy.io as sio
import os
from typing import Tuple


def prepare_pv_output(
    base_dir: str = None
) -> np.ndarray:
    """
    读取光伏出力数据

    Args:
        base_dir: 基础目录

    Returns:
        output_pv: 光伏出力 (NOFSLOTS,) MW
    """
    if base_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(os.path.dirname(script_dir))

    filename = os.path.join(base_dir, 'data_prepare', 'output_pv.mat')
    data = sio.loadmat(filename)
    output_pv = np.asarray(data['output_pv']).flatten()

    print(f"  光伏出力 shape: {output_pv.shape}")
    print(f"  光伏出力范围: [{output_pv.min():.4f}, {output_pv.max():.4f}] MW")

    return output_pv


if __name__ == "__main__":
    # 测试
    output_pv = prepare_pv_output()
    print(f"\n光伏出力数据: {output_pv}")
