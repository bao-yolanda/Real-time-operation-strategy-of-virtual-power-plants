from __future__ import annotations

from typing import Dict, Any

import numpy as np


def fast_control_implement(ctx: Dict[str, Any]) -> None:
    """快速功率分配算法：每2秒执行一次，为未来1分钟分配功率

    Args:
        ctx: 上下文字典，包含所有参数和状态变量
    """
    # ========== 输入参数 ==========
    result = ctx["result"]           # 结果字典，存储所有状态和输出
    param = ctx["param"]             # 市场参数 (价格、里程、分布等)
    param_std = ctx["param_std"]     # 资源标准化参数 (55个资源的约束)

    # 控制参数
    t_cap = ctx["t_cap"]             # 当前时间步 (1~18000秒)
    NOFDER = ctx["NOFDER"]           # 资源数量 (55个: 1 PV + 1 ES + 40 EV + 3 TCL + 10 IPP)
    NOFTCAP_ctrl = ctx["NOFTCAP_ctrl"] # 控制间隔 (30秒，即每30秒执行一次)
    delta_t = ctx["delta_t"]         # 时间步长 (0.25小时 = 15分钟)
    Signal_day = ctx["Signal_day"]   # 调频信号 (1800个点, 对应24小时, 每2秒一个点)

    # 初始化临时输出
    result["p_dis"] = []             # 放电功率 (55×30, MW)
    result["p_ch"] = []              # 充电功率 (55×30, MW)

    # ========== 当前状态变量 (从result获取) ==========
    P_DER_cur = result["P_DER_cur"]  # 当前基础功率分配 (55×1, MW)
    R_DER_cur = result["R_DER_cur"]  # 当前调频功率分配 (55×1, MW)
    E_cur = result["E_cur"]         # 当前能量状态 (55×1, MWh或等效)

    # ========== 时间计算 ==========
    cur_slot = int(np.ceil(t_cap / 1800.0))     # 当前时段编号 (1~24, 每15分钟一段)
    cur_slot_idx = cur_slot - 1                  # 当前时段索引 (0~23, 用于数组访问)

    delta_t_cap = NOFTCAP_ctrl / 1800.0          # 控制时间间隔 (30/1800 = 0.0167小时 = 1分钟)

    # ========== 调频信号 ==========
    delta = Signal_day[t_cap - 1 : t_cap - 1 + NOFTCAP_ctrl]  # 未来1分钟的调频信号 (30×1, 范围[-1,1])

    # ========== 功率需求计算 ==========
    # P_req: 55个资源 × 30个时间点的功率需求矩阵 (55×30, MW)
    # 计算逻辑: 基础功率 + 调频功率 × 信号值
    P_req = P_DER_cur[:, None] + R_DER_cur[:, None] * delta[None, :]

    # ========== 分离充放电功率 ==========
    # 将P_req分解为放电(正)和充电(负)两部分
    p_dis = 0.5 * (P_req + np.abs(P_req))   # 放电功率 (55×30, MW, P_req的正数部分)
    p_ch = 0.5 * (-P_req + np.abs(P_req))   # 充电功率 (55×30, MW, P_req的负数部分的绝对值)

    # ========== 能量约束检查 ==========
    lower_e = param_std.energy_lower_limit[:, cur_slot_idx]  # 当前时段能量下限 (55×1)
    upper_e = param_std.energy_upper_limit[:, cur_slot_idx]  # 当前时段能量上限 (55×1)

    # 如果当前能量超出约束，强制调整充放电功率
    for idx in range(NOFDER):
        if E_cur[idx] < lower_e[idx]:  # 能量过低: 强制充电
            p_dis[idx, :] = param_std.power_dis_lower_limit[idx, cur_slot_idx]  # 停止放电
            p_ch[idx, :] = param_std.power_ch_upper_limit[idx, cur_slot_idx]    # 最大充电
        if E_cur[idx] > upper_e[idx]:  # 能量过高: 强制放电
            p_ch[idx, :] = param_std.power_ch_lower_limit[idx, cur_slot_idx]    # 停止充电
            p_dis[idx, :] = param_std.power_dis_upper_limit[idx, cur_slot_idx]  # 最大放电

    # ========== 计算总充放电功率 ==========
    p_dis_sum = p_dis.sum(axis=1)  # 每个资源1分钟的总放电功率 (55×1, MW)
    p_ch_sum = p_ch.sum(axis=1)    # 每个资源1分钟的总充电功率 (55×1, MW)

    # ========== 资源参数 ==========
    theta = param_std.theta        # 能量保持率 (55×1, 通常为1, 表示无自然衰减)
    eta_ch = param_std.eta_ch      # 充电效率矩阵 (55×55, 对角矩阵)
    eta_dis = param_std.eta_dis    # 放电效率矩阵 (55×55, 对角矩阵)

    # ========== 更新能量状态 ==========
    # E_new = E_old × (1 - 自然衰减) + 充电输入 × 效率 - 放电输出 / 效率 + 外部影响
    result["E_cur"] = (
        (1.0 - delta_t_cap * (1.0 - theta)) * result["E_cur"]           # 自然衰减项
        + (eta_ch @ p_ch_sum) / 1800.0                                # 充电输入项 (转换为小时)
        - (eta_dis @ p_dis_sum) / 1800.0                               # 放电输出项 (转换为小时)
        + param_std.wOmiga[:, cur_slot_idx] * delta_t_cap              # 外部影响 (TCL的热负荷)
    )

    # ========== 记录历史数据 ==========
    result["E_rev"] = np.column_stack([result["E_rev"], result["E_cur"]])      # 能量历史 (55×t)
    result["P_alloc"] = np.column_stack([result["P_alloc"], p_dis - p_ch])    # 净功率分配历史 (55×t, 放电为正)

    # ========== 累加实际成本 (老化成本) ==========
    # cost = 放电功率 × 放电成本 + 充电功率 × 充电成本
    result["actualCost"][cur_slot_idx] += (
        np.sum(param_std.pr_dis[:, None] * p_dis)   # ES/EV的放电老化成本 ($/MWh)
        + np.sum(param_std.pr_ch[:, None] * p_ch)   # ES/EV的充电老化成本 ($/MWh)
    ) * delta_t / 1800.0  # 转换为小时单位

    # ========== 累加实际里程和能量 ==========
    P_total = np.sum(p_dis - p_ch, axis=0)  # VPP总功率 (30×1, MW)
    # 里程: 相邻功率差的绝对值之和 (反映调频响应的活跃度)
    result["actualMil"][cur_slot_idx] += np.sum(np.abs(np.diff(P_total)))
    # 能量: 功率积分 (反映净能量消耗)
    result["actualEnergy"][cur_slot_idx] += np.sum(P_total) * delta_t / 1800.0

    # ========== 输出临时结果 ==========
    result["p_dis"] = p_dis  # 当前计算的放电功率 (55×30, MW)
    result["p_ch"] = p_ch    # 当前计算的充电功率 (55×30, MW)
