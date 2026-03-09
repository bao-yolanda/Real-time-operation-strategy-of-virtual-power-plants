"""
风电数据处理
使用WT_error_gen读取风电预测值和误差
将风电误差作为调频信号
"""
import numpy as np
import sys
import os
from typing import Tuple

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from WT_error_gen import WT_sce_gen


def prepare_wind_as_regulation(
    day_wind: int = 22,
    NOFSLOTS: int = 24,
    N_wind: int = 2,
    N_samples: int = 2000,
    base_dir: str = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    准备风电数据，将风电误差作为调频信号

    Args:
        day_wind: 风电数据使用的日期
        NOFSLOTS: 时段数量
        N_wind: 风电场数量
        N_samples: 场景数量
        base_dir: 基础目录

    Returns:
        WT_pred: 风电预测值 (NOFSLOTS, N_wind)
        WT_error_scenarios: 风电误差场景 (N_samples, NOFSLOTS, N_wind)
        WT_full_scenarios: 风电完整场景 (N_samples, NOFSLOTS, N_wind)
        hourly_Distribution: 小时级误差分布 (NOFSLOTS, NOFSCEN)
        hourly_Mileage: 小时级里程 (NOFSLOTS,)
        d_s: 场景信号值 (NOFSCEN,)
        Signal_day: 当日信号 (signal_length,)
    """
    if base_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(os.path.dirname(script_dir))

    print("="*70)
    print("风电数据准备 - 将风电误差作为调频信号")
    print("="*70)
    print(f"\n配置:")
    print(f"  风电场数量 (N_wind): {N_wind}")
    print(f"  时段数 (NOFSLOTS): {NOFSLOTS}")
    print(f"  场景数 (N_samples): {N_samples}")

    # ========== 读取风电预测和误差 ==========
    print("\n" + "-"*70)
    print("步骤1: 读取风电预测和误差数据...")
    print("-"*70)

    # 使用WT_error_gen函数读取数据
    WT_pred_full, WT_error_scenarios_full, WT_full_scenarios_full = WT_sce_gen(N_wind, N_samples, seed=0)

    # WT_pred_full shape: (T_total, N_wind), T_total是总时长
    # 我们需要取指定日期的数据
    # 假设每天24小时，我们需要第day_wind天的数据
    T_per_day = 24
    start_idx = (day_wind - 1) * T_per_day
    end_idx = start_idx + NOFSLOTS

    # 检查数据长度是否足够
    if end_idx > WT_pred_full.shape[0]:
        print(f"警告: 数据长度不足，使用所有可用数据")
        end_idx = WT_pred_full.shape[0]
        start_idx = max(0, end_idx - NOFSLOTS)

    # 提取指定日期的数据
    WT_pred = WT_pred_full[start_idx:end_idx, :]  # (NOFSLOTS, N_wind)
    WT_error_scenarios = WT_error_scenarios_full[:, start_idx:end_idx, :]  # (N_samples, NOFSLOTS, N_wind)
    WT_full_scenarios = WT_full_scenarios_full[:, start_idx:end_idx, :]  # (N_samples, NOFSLOTS, N_wind)

    print(f"  风电预测数据 shape: {WT_pred.shape}")
    print(f"  风电误差场景 shape: {WT_error_scenarios.shape}")
    print(f"  风电完整场景 shape: {WT_full_scenarios.shape}")

    # 对所有风电场求和
    WT_pred_total = np.sum(WT_pred, axis=1)  # (NOFSLOTS,)
    WT_error_total = np.sum(WT_error_scenarios, axis=2)  # (N_samples, NOFSLOTS)
    WT_full_total = np.sum(WT_full_scenarios, axis=2)  # (N_samples, NOFSLOTS)

    print(f"  总风电预测 (所有风电场): {WT_pred_total.shape}")
    print(f"  风电预测范围: [{WT_pred_total.min():.4f}, {WT_pred_total.max():.4f}]")
    print(f"  误差范围: [{WT_error_total.min():.4f}, {WT_error_total.max():.4f}]")

    # ========== 计算误差的离散分布 ==========
    print("\n" + "-"*70)
    print("步骤2: 计算风电误差的离散分布...")
    print("-"*70)

    # 离散化参数
    granularity = 0.1  # 离散化步长
    n_intervals = int(2 / granularity) + 2  # 场景数: [-1, ..., -0.1, 0, 0.1, ..., 1]

    # 生成场景信号值 d_s
    d_s_values = np.zeros(n_intervals)
    d_s_values[0] = -1.0
    for i in range(1, n_intervals - 1):
        d_s_values[i] = -1 + i * granularity
    d_s_values[-1] = 1.0

    print(f"  场景数 (NOFSCEN): {n_intervals}")
    print(f"  d_s values 范围: [{d_s_values.min():.2f}, {d_s_values.max():.2f}]")

    # 计算每小时的误差分布
    hourly_Distribution = np.zeros((NOFSLOTS, n_intervals))
    hourly_Mileage = np.zeros(NOFSLOTS)

    # 对每个小时
    for hour in range(NOFSLOTS):
        # 获取该小时所有场景的误差
        hour_errors = WT_error_total[:, hour]  # (N_samples,)

        # 计算每个场景区间的频率
        for i in range(n_intervals):
            if i == 0:  # 第一个区间: [-inf, -0.95]
                count = np.sum(hour_errors < -0.95)
            elif i == n_intervals - 1:  # 最后一个区间: [0.95, inf]
                count = np.sum(hour_errors >= 0.95)
            else:  # 中间区间
                lower_bound = d_s_values[i] - granularity/2
                upper_bound = d_s_values[i] + granularity/2
                count = np.sum((hour_errors >= lower_bound) & (hour_errors < upper_bound))

            hourly_Distribution[hour, i] = count

        # 归一化
        sum_dist = np.sum(hourly_Distribution[hour, :])
        if sum_dist > 0:
            hourly_Distribution[hour, :] = hourly_Distribution[hour, :] / sum_dist
        else:
            # 如果没有数据，使用均匀分布
            hourly_Distribution[hour, :] = np.ones(n_intervals) / n_intervals

        # 计算里程（相邻误差差的绝对值）
        # 使用场景平均来计算预期里程
        avg_error = np.mean(hour_errors)
        if hour > 0:
            prev_avg_error = np.mean(WT_error_total[:, hour-1])
            mileage = np.abs(avg_error - prev_avg_error)
        else:
            mileage = 0.0
        hourly_Mileage[hour] = mileage

    print(f"  hourly_Distribution shape: {hourly_Distribution.shape}")
    print(f"  分布验证: {np.allclose(hourly_Distribution.sum(axis=1), 1.0)}")

    # ========== 生成当日信号 ==========
    print("\n" + "-"*70)
    print("步骤3: 生成当日风电误差信号...")
    print("-"*70)

    # 使用第一个场景作为当日信号
    Signal_day = WT_error_total[0, :]  # (NOFSLOTS,)

    # 扩展到2秒分辨率（每分钟60个点）
    # 假设每60秒=1800个点
    signal_length = NOFSLOTS * 1800
    Signal_day_extended = np.zeros(signal_length)

    for hour in range(NOFSLOTS):
        start_idx = hour * 1800
        end_idx = (hour + 1) * 1800
        Signal_day_extended[start_idx:end_idx] = Signal_day[hour]

    print(f"  当日信号长度: {signal_length}")
    print(f"  信号范围: [{Signal_day.min():.4f}, {Signal_day.max():.4f}]")

    # ========== 总结 ==========
    print("\n" + "="*70)
    print("风电数据准备完成")
    print("="*70)
    print(f"\n输出:")
    print(f"  WT_pred: 风电预测值 ({WT_pred.shape})")
    print(f"  WT_error_scenarios: 风电误差场景 ({WT_error_scenarios.shape})")
    print(f"  WT_full_scenarios: 风电完整场景 ({WT_full_scenarios.shape})")
    print(f"  hourly_Distribution: 小时误差分布 ({hourly_Distribution.shape})")
    print(f"  hourly_Mileage: 小时里程 ({hourly_Mileage.shape})")
    print(f"  d_s: 场景信号值 ({d_s_values.shape})")
    print(f"  Signal_day: 当日信号 ({Signal_day_extended.shape})")

    return (WT_pred, WT_error_scenarios, WT_full_scenarios,
            hourly_Distribution, hourly_Mileage, d_s_values, Signal_day_extended)


def prepare_wind_power_output(
    day_wind: int = 22,
    NOFSLOTS: int = 24,
    N_wind: int = 2,
    base_dir: str = None
) -> np.ndarray:
    """
    准备风电功率输出（替代PV输出）

    Args:
        day_wind: 风电数据日期
        NOFSLOTS: 时段数
        N_wind: 风电场数量
        base_dir: 基础目录

    Returns:
        wind_power: 风电功率输出 (NOFSLOTS,)，单位：MW
    """
    if base_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(os.path.dirname(script_dir))

    # 读取风电预测
    WT_pred_full, _, _ = WT_sce_gen(N_wind, N_samples=1, seed=0)

    # 提取指定日期
    T_per_day = 24
    start_idx = (day_wind - 1) * T_per_day
    end_idx = start_idx + NOFSLOTS

    if end_idx > WT_pred_full.shape[0]:
        end_idx = WT_pred_full.shape[0]
        start_idx = max(0, end_idx - NOFSLOTS)

    WT_pred = WT_pred_full[start_idx:end_idx, :]

    # 对所有风电场求和并转换为MW
    # 假设归一化后的值乘以风电场容量
    wind_capacity = 10.0  # MW，单个风电场容量
    wind_power = np.sum(WT_pred, axis=1) * wind_capacity

    print(f"风电功率输出 shape: {wind_power.shape}")
    print(f"功率范围: [{wind_power.min():.2f}, {wind_power.max():.2f}] MW")

    return wind_power


if __name__ == "__main__":
    # 测试风电数据处理
    (WT_pred, WT_error_scenarios, WT_full_scenarios,
     hourly_Distribution, hourly_Mileage, d_s, Signal_day) = prepare_wind_as_regulation(
        day_wind=22,
        NOFSLOTS=24,
        N_wind=2,
        N_samples=1000
    )

    print(f"\n测试通过!")
    print(f"风电预测（前5小时）: {WT_pred[:5, :]}")
    print(f"d_s（前5个场景）: {d_s[:5]}")
