"""
调频信号处理
读取RegD信号数据并计算离散分布
"""
import numpy as np
import openpyxl
import os
from typing import Tuple, Dict
import pandas as pd

def prepare_regd_distribution(
    day_reg: int = 22,
    NOFSLOTS: int = 24,
    granularity: float = 0.1,
    nofHisDays: int = 14,
    base_dir: str = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取并处理RegD信号数据

    Args:
        day_reg: 调频信号日期 (如 22)
        NOFSLOTS: 时段数量
        granularity: 离散化粒度
        nofHisDays: 历史天数
        base_dir: 基础目录

    Returns:
        hourly_Distribution: 小时级分布 (NOFSLOTS, NOFSCEN)
        hourly_Mileage: 小时级里程 (NOFSLOTS,)
        d_s: 场景信号值 (NOFSCEN,)
        Signal_day: 当日信号 (signal_length,)
    """
    if base_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(os.path.dirname(script_dir))

    # ========== 读取RegD信号数据 ==========
    filename = os.path.join(base_dir, 'data_prepare', '07 2020.xlsx')
    sheet = 'Dynamic'

    df = pd.read_excel(filename, header=None, sheet_name=sheet, usecols=range(1, 32))  # 读取B-AF列（31列）

    # 转换为数值类型，跳过第0行（表头）
    Signals = np.zeros((43201, 31))
    for col_idx in range(df.shape[1]):
        col_data = df.iloc[1:43202, col_idx].values  # 第1行到第43201行（0-based）
        try:
            Signals[:, col_idx] = col_data.astype(float)
        except:
            Signals[:, col_idx] = 0.0

    Signals = Signals[:43200, :]  # (43200, 31)

    Signals[Signals < -1] = -1
    Signals[Signals > 1] = 1

    print(f"  信号数据 shape: {Signals.shape}")
    print(f"  信号范围: [{Signals.min():.4f}, {Signals.max():.4f}]")

    # ========== 当日信号数据 ==========
    Signal_day = Signals[:, col_idx]

    # ========== 计算离散化参数 ==========
    diff = granularity  # 离散化步长
    n_intervals = int(2 / diff) + 2  # 场景数: [-1, ..., -diff, 0, diff, ..., 1]

    # 生成场景信号值 d_s
    # 从 -1 到 1，包含端点
    d_s_values = np.zeros(n_intervals)
    d_s_values[0] = -1.0  # 第一个场景：-1
    for i in range(1, n_intervals - 1):
        d_s_values[i] = -1 + i * diff
    d_s_values[-1] = 1.0  # 最后一个场景：1

    print(f"  场景数 (NOFSCEN): {n_intervals}")
    print(f"  d_s values: {d_s_values}")

    # ========== 按小时计算分布 ==========
    hourly_Distribution = np.zeros((NOFSLOTS, n_intervals))
    hourly_Mileage = np.zeros(NOFSLOTS)

    for hour in range(NOFSLOTS):

        start_idx = hour * 1800
        end_idx = (hour + 1) * 1800

        Distributions = np.zeros((nofHisDays, n_intervals))

        day_idx_offset = 0
        for day_idx in range(day_reg - nofHisDays, day_reg):
            signals = Signals[start_idx:end_idx, day_idx]  # (1800,)

            Distribution = np.zeros(n_intervals)

            for t_cap in range(len(signals)):
                signal_val = signals[t_cap]

                if signal_val >= 0:  # 向上频率调节
                    if signal_val > 0.9999:  # 考虑为1
                        s_idx = n_intervals - 1
                    else:
                        s_idx = int(np.ceil(signal_val / diff) + 1 / diff + 1)
                else:  # 向下频率调节
                    if signal_val < -0.9999:  # 考虑为-1
                        s_idx = 0
                    else:
                        # s_idx = floor(signal / diff) + 1/diff + 2
                        s_idx = int(np.floor(signal_val / diff) + 1 / diff + 2)

                # 调试：只打印前几个信号
                if hour == 0 and day_idx_offset == 0 and t_cap < 5:
                    print(f"    信号[{t_cap}]={signal_val:.4f} -> s_idx={s_idx}")

                Distribution[s_idx] += 1

            # 计算频率
            sum_dist = np.sum(Distribution)
            if sum_dist > 0:
                Distribution = Distribution / sum_dist
            else:
                # 如果该小时没有数据，使用均匀分布
                Distribution = np.ones(n_intervals) / n_intervals

            if hour == 0:
                print(f"    第{day_idx_offset+1}天（day_idx={day_idx}）分布: sum={sum_dist:.0f}, 归一化后sum={Distribution.sum():.6f}")

            Distributions[day_idx_offset, :] = Distribution
            day_idx_offset += 1

        avg_Distribution = np.mean(Distributions, axis=0)
        hourly_Distribution[hour, :] = avg_Distribution

        # 计算里程
        Mileage_values = np.zeros(nofHisDays)
        day_idx_offset = 0
        for day_idx in range(day_reg - nofHisDays, day_reg):

            signals = Signals[start_idx:end_idx, day_idx]
            mileage = np.sum(np.abs(np.diff(signals)))
            Mileage_values[day_idx_offset] = mileage
            day_idx_offset += 1

        hourly_Mileage[hour] = np.mean(Mileage_values)

    print(f"  hourly_Distribution shape: {hourly_Distribution.shape}")
    print(f"  每小时概率和验证: {np.allclose(hourly_Distribution.sum(axis=1), 1.0)}")

    # 调试：检查第一小时的分布
    print(f"  第1小时分布示例: {hourly_Distribution[0, :]}")
    print(f"  第1小时分布和: {hourly_Distribution[0, :].sum():.6f}")

    return hourly_Distribution, hourly_Mileage, d_s_values, Signal_day


if __name__ == "__main__":
    hourly_Dist, hourly_Mileage, d_s, Signal_day = prepare_regd_distribution()
    print(f"\n测试通过!")
    print(f"d_s: {d_s}")
    print(f"hourly_Mileage 范围: [{hourly_Mileage.min():.2f}, {hourly_Mileage.max():.2f}]")
