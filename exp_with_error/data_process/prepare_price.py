"""
价格数据准备
读取调频市场价格和系统能量价格
"""
import numpy as np
import openpyxl
import os
from typing import Tuple, Dict


def prepare_price_data(
    day_price: int = 21,
    hour_init: int = 0,
    NOFSLOTS: int = 24,
    base_dir: str = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    读取市场价格数据

    Args:
        day_price: 价格日期 (如 21)
        hour_init: 起始小时
        NOFSLOTS: 时段数量
        base_dir: 基础目录

    Returns:
        price_reg: 调频价格 (NOFSLOTS, 2) - [容量价格, 里程价格]
        price_e: 能量价格 (NOFSLOTS,)
        start_row: 起始行号
    """
    if base_dir is None:
        # 默认使用 data_prepare 目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(os.path.dirname(script_dir))

    # ========== 读取调频市场价格数据 ==========
    filename = os.path.join(base_dir, 'data_prepare', 'regulation_market_results.xlsx')
    sheet = 'regulation_market_results'
    start_row = (day_price - 1) * 24 + hour_init + 2

    wb = openpyxl.load_workbook(filename)
    ws = wb[sheet]

    # 读取容量价格和里程价格 (G列和H列)
    price_reg = np.zeros((NOFSLOTS, 2))
    for i in range(NOFSLOTS):
        price_reg[i, 0] = ws.cell(row=start_row + i, column=7).value  # G列: 容量价格
        price_reg[i, 1] = ws.cell(row=start_row + i, column=8).value  # H列: 里程价格

    wb.close()

    # ========== 读取系统能量价格数据 ==========
    filename = os.path.join(base_dir, 'data_prepare', 'rt_hrl_lmps.xlsx')
    sheet = 'rt_hrl_lmps'

    wb = openpyxl.load_workbook(filename)
    ws = wb[sheet]

    # 读取能量价格 (I列)
    price_e = np.zeros(NOFSLOTS)
    for i in range(NOFSLOTS):
        price_e[i] = ws.cell(row=start_row + i, column=9).value  # I列: 能量价格

    wb.close()

    print(f"  调频价格 shape: {price_reg.shape}")
    print(f"  容量价格范围: [{price_reg[:, 0].min():.2f}, {price_reg[:, 0].max():.2f}] $/MW")
    print(f"  里程价格范围: [{price_reg[:, 1].min():.2f}, {price_reg[:, 1].max():.2f}] $/MW")
    print(f"  能量价格 shape: {price_e.shape}")
    print(f"  能量价格范围: [{price_e.min():.2f}, {price_e.max():.2f}] $/MWh")

    return price_reg, price_e, start_row


if __name__ == "__main__":
    # 测试
    price_reg, price_e, start_row = prepare_price_data(day_price=21)
    print(f"\n测试通过! 起始行: {start_row}")
