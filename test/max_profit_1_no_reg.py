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


def max_profit_1(ctx: Dict[str, Any]) -> None:
    """日前初始优化：在仿真开始时优化全天的充放电策略

    使用CVXPY求解混合整数规划问题，优化VPP的能量套利策略
    不参与调频市场，只进行能量市场套利
    """
    # ========== 输入参数 ==========
    param = ctx["param"]         # 市场参数 (能量价格)
    param_std = ctx["param_std"] # 资源标准化参数
    result = ctx["result"]       # 结果字典

    # 控制参数
    NOFSLOTS = ctx["NOFSLOTS"]           # 时间段数 (24小时)
    NOFDER = ctx["NOFDER"]               # 资源数量 (42: PV+ES+EV)
    delta_t = ctx["delta_t"]             # 时间步长 (1小时)

    # 调试：打印参数形状
    print(f"Debug: NOFDER = {NOFDER}")
    print(f"Debug: eta_ch shape = {param_std['eta_ch'].shape}")
    print(f"Debug: eta_dis shape = {param_std['eta_dis'].shape}")

    # ========== 决策变量 ==========
    # 功率决策
    P_dis = cp.Variable((NOFDER, NOFSLOTS))  # 放电功率 (42×24, MW)
    P_ch = cp.Variable((NOFDER, NOFSLOTS))    # 充电功率 (42×24, MW)
    E = cp.Variable((NOFDER, NOFSLOTS + 1))    # 能量状态 (42×25, MWh)

    # PV 出力 (已知参数，不是变量)
    PV_output = param_std['power_dis_upper_limit'][0, :]  # PV 出力 (1×24, MW)

    # ========== 市场价格参数 ==========
    price_e = param['price_e']  # 实时能量价格 (24×1, $/MWh)

    # ========== 目标函数：最大化利润 ==========
    # 利润 = 放电收益 - 充电成本 - 能量市场净支出

    # 净放电功率 = P_dis - P_ch
    net_power = cp.sum(P_dis, axis=0) - cp.sum(P_ch, axis=0)

    # PV 出力增加净功率
    net_power_with_pv = net_power + PV_output

    # 能量市场收益 = net_power_with_pv × price_e
    energy_revenue = price_e.T @ net_power_with_pv

    # 充放电老化成本
    degradation_cost = cp.sum(
        cp.multiply(param_std['pr_dis'][:, None], P_dis)  # 放电老化成本
        + cp.multiply(param_std['pr_ch'][:, None], P_ch),  # 充电老化成本
    )

    # 总利润 = 能量市场收益 - 老化成本
    Profit = energy_revenue - degradation_cost

    # ========== 约束条件 ==========
    constraints = []

    # 1. 初始能量约束
    constraints += [E[:, 0] == param_std['energy_init']]

    # 2. 功率上下限约束
    constraints += [
        P_dis >= param_std['power_dis_lower_limit'],  # 放电功率下限
        P_ch >= param_std['power_ch_lower_limit'],    # 充电功率下限
        P_dis <= param_std['power_dis_upper_limit'],  # 放电功率上限
        P_ch <= param_std['power_ch_upper_limit'],    # 充电功率上限
    ]

    # 3. 能量上下限约束
    constraints += [param_std['energy_lower_limit'] <= E[:, 1:]]  # 能量下限
    constraints += [E[:, 1:] <= param_std['energy_upper_limit']]  # 能量上限

    # 4. 能量动态方程
    # E[t+1] = E[t] × theta + eta_ch × P_ch × delta_t - P_dis / eta_dis × delta_t
    # theta = 1 (PV, ES, EV 都没有衰减)
    eta_ch = param_std['eta_ch']  # 充电效率矩阵 (42×42)
    eta_dis = param_std['eta_dis']  # 放电效率矩阵 (42×42)

    constraints += [
        E[:, 1:] == E[:, :-1]                          # theta=1，无衰减
        + (eta_ch @ P_ch) * delta_t         # 充电输入项
        - (eta_dis @ P_dis) * delta_t        # 放电输出项
    ]

    # 5. 最终能量约束 (EV 需要达到目标能量)
    constraints += [E[:, -1] >= param_std['energy_end']]

    # 6. PV 特殊约束
    # PV 只能放电 (发电)，不能充电
    constraints += [P_ch[0, :] == 0]  # PV 充电功率为 0
    constraints += [P_dis[0, :] == PV_output]  # PV 放电功率等于其出力

    # ========== 求解优化问题 ==========
    objective = cp.Maximize(Profit)
    problem = cp.Problem(objective, constraints)
    solver_name = _choose_solver("GUROBI")  # 优先使用Gurobi求解器
    problem.solve(solver=solver_name, verbose=False)

    # ========== 检查求解状态 ==========
    ok = problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    if ok:
        print("slot 1: energy optimization ok")
        print(f"  Total profit: ${Profit.value:.2f}")
        print(f"  Energy revenue: ${energy_revenue.value:.2f}")
        print(f"  Degradation cost: ${degradation_cost.value:.2f}")
    else:
        print(f"slot 1: energy optimization failed, status: {problem.status}")

    # ========== 存储初始优化结果 (全天最优解) ==========
    result["P_dis_init"] = P_dis.value      # 初始放电功率 (42×24, MW)
    result["P_ch_init"] = P_ch.value        # 初始充电功率 (42×24, MW)
    result["E_init"] = E.value              # 初始能量状态 (42×25)

    # ========== 存储当前时段状态 (用于实时控制) ==========
    result["P_dis_cur"] = P_dis.value[:, 0]  # 当前时段放电功率 (42×1, MW)
    result["P_ch_cur"] = P_ch.value[:, 0]    # 当前时段充电功率 (42×1, MW)
    result["E_cur"] = E.value[:, 0]         # 当前能量状态 (42×1)

    # ========== 存储历史记录 ==========
    result["P_dis_rev"] = P_dis.value.copy()  # 放电功率历史 (42×24)
    result["P_ch_rev"] = P_ch.value.copy()    # 充电功率历史 (42×24)
    result["E_rev"] = result["E_cur"].reshape(-1, 1)  # 能量历史 (42×1, 初始状态)
