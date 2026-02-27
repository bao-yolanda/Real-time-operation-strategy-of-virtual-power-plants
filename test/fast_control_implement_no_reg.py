from __future__ import annotations

from typing import Dict, Any

import numpy as np


def fast_control_implement(ctx: Dict[str, Any]) -> None:
    """实时控制：在每个控制时刻执行实际控制动作

    不参与调频市场，只执行充放电决策
    """
    # ========== 输入参数 ==========
    param_std = ctx["param_std"]
    result = ctx["result"]

    t_cap = ctx["t_cap"]  # 当前时刻 (从1开始，单位：2秒)
    NOFDER = ctx["NOFDER"]
    delta_t = ctx["delta_t"]
    delta_t_req = ctx["delta_t_req"]

    # 计算当前时段索引
    t_slot = int((t_cap - 1) / 1800)  # 0-based index

    # ========== 执行充放电决策 ==========
    # 获取当前时段的充放电功率决策
    P_dis = result["P_dis_cur"]  # 放电功率 (42×1, MW)
    P_ch = result["P_ch_cur"]    # 充电功率 (42×1, MW)

    # ========== 更新能量状态 ==========
    E_current = result["E_cur"]  # 当前能量 (42×1, MWh)
    E_new = E_current.copy()

    # 能量动态方程: E_new = E + eta_ch × P_ch × delta_t - P_dis / eta_dis × delta_t
    eta_ch = param_std['eta_ch']
    eta_dis = param_std['eta_dis']

    E_new = E_current + (eta_ch @ P_ch) * delta_t - (eta_dis @ P_dis) * delta_t

    # 更新当前能量
    result["E_cur"] = E_new

    # ========== 存储控制结果 ==========
    # 净放电功率
    net_power = np.sum(P_dis) - np.sum(P_ch)

    # PV 出力
    PV_output = param_std['power_dis_upper_limit'][0, t_slot] if t_slot < 24 else 0
    net_power_with_pv = net_power + PV_output

    # 存储能量历史
    if "E_rev" in result:
        result["E_rev"] = np.hstack([result["E_rev"], E_new.reshape(-1, 1)])

    # ========== 计算实际成本 ==========
    # 能量市场净支出
    if t_slot < 24:
        price_e = ctx["param"]['price_e'][t_slot]
        energy_cost = -net_power_with_pv * price_e  # 负数表示支出
    else:
        energy_cost = 0

    # 充放电老化成本
    degradation_cost = np.sum(
        param_std['pr_dis'] * P_dis
        + param_std['pr_ch'] * P_ch
    )

    # 存储实际成本
    if "actualCost" in result and len(result["actualCost"]) > t_slot:
        result["actualCost"][t_slot] += energy_cost + degradation_cost

    # ========== 存储功率分配历史 ==========
    if "P_alloc" in result and result["P_alloc"].shape[1] <= t_slot:
        P_alloc = np.zeros((NOFDER, 1))
        P_alloc[:, 0] = P_dis - P_ch  # 净放电功率
        result["P_alloc"] = np.hstack([result["P_alloc"], P_alloc])
