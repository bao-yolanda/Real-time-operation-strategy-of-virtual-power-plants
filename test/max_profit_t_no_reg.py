from __future__ import annotations

from typing import Dict, Any

import cvxpy as cp
import numpy as np


def _choose_solver(preferred: str) -> str:
    installed = cp.installed_solvers()
    if preferred in installed:
        return preferred
    for fallback in ["ECOS", "OSQP", "SCS"]:
        if fallback in installed:
            return fallback
    return "SCS"


def max_profit_t(ctx: Dict[str, Any]) -> None:
    """滚动优化：在每个更新时刻重新优化剩余时段的策略

    使用CVXPY求解混合整数规划问题，更新剩余时段的充放电策略
    不参与调频市场，只进行能量市场套利
    """
    # ========== 输入参数 ==========
    param = ctx["param"]         # 市场参数 (能量价格)
    param_std = ctx["param_std"] # 资源标准化参数
    result = ctx["result"]       # 结果字典

    # 控制参数
    t_cap = ctx["t_cap"]        # 当前时刻 (从1开始，单位：2秒)
    delta_t_rest = ctx.get("delta_t_rest", 1.0)  # 剩余时段长度 (小时)
    NOFSLOTS = ctx["NOFSLOTS"]   # 时间段数 (24小时)
    NOFDER = ctx["NOFDER"]       # 资源数量 (42: PV+ES+EV)
    delta_t = ctx["delta_t"]     # 时间步长 (1小时)

    # 计算当前时段索引
    t_slot = int((t_cap - 1) / 1800)  # 0-based index

    # ========== 决策变量 ==========
    # 只优化剩余时段 (t_slot 到 NOFSLOTS-1)
    remaining_slots = NOFSLOTS - t_slot

    if remaining_slots <= 0:
        print(f"slot {t_slot + 1}: no remaining slots, skip optimization")
        return

    # 功率决策
    P_dis = cp.Variable((NOFDER, remaining_slots))  # 放电功率 (42×remaining, MW)
    P_ch = cp.Variable((NOFDER, remaining_slots))    # 充电功率 (42×remaining, MW)
    E = cp.Variable((NOFDER, remaining_slots + 1))    # 能量状态 (42×remaining+1, MWh)

    # PV 出力 (已知参数)
    PV_output = param_std['power_dis_upper_limit'][0, t_slot:]  # PV 出力 (1×remaining, MW)

    # ========== 市场价格参数 ==========
    price_e = param['price_e'][t_slot:]  # 剩余时段能量价格 (remaining×1, $/MWh)

    # ========== 目标函数：最大化剩余利润 ==========
    # 净放电功率 = P_dis - P_ch
    net_power = cp.sum(P_dis, axis=0) - cp.sum(P_ch, axis=0)
    net_power_with_pv = net_power + PV_output

    # 能量市场收益
    energy_revenue = price_e.T @ net_power_with_pv

    # 充放电老化成本
    degradation_cost = cp.sum(
        cp.multiply(param_std['pr_dis'][:, None], P_dis)
        + cp.multiply(param_std['pr_ch'][:, None], P_ch),
    )

    # 总利润
    Profit = energy_revenue - degradation_cost

    # ========== 约束条件 ==========
    constraints = []

    # 1. 初始能量约束 (使用当前实际能量)
    E_current = result["E_cur"]  # 当前能量状态 (42×1)
    constraints += [E[:, 0] == E_current]

    # 2. 功率上下限约束
    constraints += [
        P_dis >= param_std['power_dis_lower_limit'][:, t_slot:],
        P_ch >= param_std['power_ch_lower_limit'][:, t_slot:],
        P_dis <= param_std['power_dis_upper_limit'][:, t_slot:],
        P_ch <= param_std['power_ch_upper_limit'][:, t_slot:],
    ]

    # 3. 能量上下限约束
    constraints += [param_std['energy_lower_limit'][:, t_slot:] <= E[:, 1:]]
    constraints += [E[:, 1:] <= param_std['energy_upper_limit'][:, t_slot:]]

    # 4. 能量动态方程
    eta_ch = param_std['eta_ch']
    eta_dis = param_std['eta_dis']

    constraints += [
        E[:, 1:] == E[:, :-1]
        + (eta_ch @ P_ch) * delta_t
        - (eta_dis @ P_dis) * delta_t
    ]

    # 5. 最终能量约束
    constraints += [E[:, -1] >= param_std['energy_end']]

    # 6. PV 特殊约束
    constraints += [P_ch[0, :] == 0]
    constraints += [P_dis[0, :] == PV_output]

    # ========== 求解优化问题 ==========
    objective = cp.Maximize(Profit)
    problem = cp.Problem(objective, constraints)
    solver_name = _choose_solver("GUROBI")
    problem.solve(solver=solver_name, verbose=False)

    # ========== 检查求解状态 ==========
    ok = problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    if ok:
        print(f"slot {t_slot + 1}: rolling optimization ok")
    else:
        print(f"slot {t_slot + 1}: rolling optimization failed, status: {problem.status}")

    # ========== 更新当前时段结果 ==========
    if P_dis.value is not None and P_ch.value is not None:
        result["P_dis_cur"] = P_dis.value[:, 0]
        result["P_ch_cur"] = P_ch.value[:, 0]

    # ========== 存储历史记录 ==========
    if "P_dis_rev" in result and result["P_dis_rev"].shape[1] >= t_slot + 1:
        # 更新历史记录
        result["P_dis_rev"][:, t_slot:] = P_dis.value
        result["P_ch_rev"][:, t_slot:] = P_ch.value
