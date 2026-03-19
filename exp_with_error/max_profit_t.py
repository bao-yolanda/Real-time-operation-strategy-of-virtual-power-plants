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
    """实时滚动投标优化：每15分钟执行一次，重新优化剩余时段的投标策略

    考虑当前实际状态，滚动优化剩余时段的投标
    """
    # ========== 输入参数 ==========
    param = ctx["param"]         # 市场参数 (价格、里程、分布等)
    param_std = ctx["param_std"] # 资源标准化参数 (55个资源的约束)
    result = ctx["result"]       # 结果字典

    # 控制参数
    NOFSLOTS = ctx["NOFSLOTS"]           # 总时间段数 (96, 每15分钟一段)
    NOFDER = ctx["NOFDER"]               # 资源数量 (55)
    NOFSCEN = ctx["NOFSCEN"]             # 场景数
    delta_t = ctx["delta_t"]             # 时间步长 (0.25小时 = 15分钟)
    delta_t_req = ctx["delta_t_req"]     # 响应时间间隔 (0.5小时)
    delta_t_rest = ctx["delta_t_rest"]   # 当前时段剩余时间 (小时)
    t_cap = ctx["t_cap"]                 # 当前时间步 (1~18000秒)

    # ========== 时间索引计算 ==========
    cur_slot = int(np.ceil(t_cap / 1800.0))     # 当前时段编号 (1~24)
    cur_slot_idx = cur_slot - 1                  # 当前时段索引 (0~23)
    rest_slots = NOFSLOTS - cur_slot              # 剩余时段数

    # ========== 获取当前实际状态 ==========
    Bid_R_cur = result["Bid_R_rev"][cur_slot_idx]  # 当前时段调频投标 (MW)
    Bid_P_cur = result["Bid_P_rev"][cur_slot_idx]  # 当前时段基础功率投标 (MW)
    E_cur = result["E_cur"]                         # 当前能量状态 (55×1)
    P_DER_cur = result["P_DER_rev"][:, cur_slot_idx]  # 当前各资源基础功率 (55×1, MW)
    R_DER_cur = result["R_DER_rev"][:, cur_slot_idx]  # 当前各资源调频功率 (55×1, MW)
    result["P_DER_cur"] = P_DER_cur  # 存储当前基础功率 (用于快速控制)
    result["R_DER_cur"] = R_DER_cur  # 存储当前调频功率 (用于快速控制)

    # ========== 决策变量 (优化剩余时段) ==========
    Bid_P = cp.Variable(rest_slots)         # 剩余时段基础功率投标
    Bid_R = cp.Variable(rest_slots)         # 剩余时段调频功率投标
    R_DER = cp.Variable((NOFDER, rest_slots))  # 剩余时段各资源调频功率 (55×rest_slots)
    P_DER = cp.Variable((NOFDER, rest_slots))  # 剩余时段各资源基础功率 (55×rest_slots)

    # 场景相关变量 (考虑当前时段和剩余时段)
    P_dis = cp.Variable((NOFDER, rest_slots + 1, NOFSCEN))  # 放电功率 (55×(rest+1)×场景数)
    P_ch = cp.Variable((NOFDER, rest_slots + 1, NOFSCEN))    # 充电功率 (55×(rest+1)×场景数)
    E = cp.Variable((NOFDER, rest_slots + 2))            # 能量状态 (55×(rest+2))

    # 软约束变量 (允许能量约束松弛，以获得可行解)
    delta_E1 = cp.Variable((NOFDER, rest_slots + 2))  # 能量下限松弛变量
    delta_E2 = cp.Variable((NOFDER, rest_slots + 2))  # 能量上限松弛变量
    delta_E3 = cp.Variable((NOFDER, rest_slots))      # 向下调频约束松弛
    delta_E4 = cp.Variable((NOFDER, rest_slots))      # 向上调频约束松弛
    Cost_perf = cp.Variable((rest_slots + 1, NOFSCEN))  # 老化成本

    # ========== 获取剩余时段的市场参数 ==========
    price_e = param.price_e[cur_slot_idx + 1 :]                      # 剩余时段能量价格
    price_reg = param.price_reg[cur_slot_idx + 1 :]                  # 剩余时段调频价格
    hourly_mileage = param.hourly_Mileage[cur_slot_idx + 1 :]       # 剩余时段里程
    hourly_distribution = param.hourly_Distribution[cur_slot_idx + 1 :, :]  # 剩余时段信号分布

    # ========== 收益计算 (仅剩余时段) ==========
    # 1. 里程收益
    reg_mileage = price_reg[:, 1] * hourly_mileage
    # 2. 调频能量收益
    reg_energy = cp.multiply(hourly_distribution @ param.d_s, price_e)
    # 3. 总利润 (剩余时段)
    Profit = (
        price_e @ Bid_P                                  # 能量收益
        + price_reg[:, 0] @ Bid_R * param.s_perf         # 容量收益
        + reg_mileage @ Bid_R * param.s_perf             # 里程收益
        + reg_energy @ Bid_R                              # 调频能量收益
        - cp.sum(cp.multiply(hourly_distribution, Cost_perf[1:, :]))  # 老化成本
    )
    Profit = Profit * delta_t

    # ========== 减去当前时段的收益 (因为已经执行) ==========
    cur_dist = param.hourly_Distribution[cur_slot_idx, :]  # 当前时段信号分布
    # 1. 当前时段的调频能量收益
    Profit = Profit - (cur_dist @ param.d_s * param.price_e[cur_slot_idx]) * Bid_R_cur * delta_t_rest
    # 2. 当前时段的老化成本
    Profit = Profit - cp.sum(cp.multiply(cur_dist, Cost_perf[0, :])) * delta_t_rest

    # ========== 软约束惩罚项 ==========
    M = ctx["M"]  # 大M常数 (1e6), 用于惩罚约束松弛
    Profit = Profit - M * (
        cp.sum(delta_E1) + cp.sum(delta_E2) + cp.sum(delta_E3) + cp.sum(delta_E4)
    )

    # ========== 约束条件 ==========
    constraints = []

    # 1. 初始能量约束 (等于当前实际状态)
    constraints += [E[:, 0] == E_cur]

    # 2. 功量平衡约束
    d_s = param.d_s.reshape((1, 1, NOFSCEN))
    # 剩余时段: P_dis - P_ch = P_DER + R_DER × 信号值
    constraints += [
        P_dis[:, 1:, :] - P_ch[:, 1:, :]
        == cp.reshape(P_DER, (NOFDER, rest_slots, 1))
        + cp.multiply(cp.reshape(R_DER, (NOFDER, rest_slots, 1)), d_s)
    ]
    # 当前时段剩余时间: P_dis - P_ch = P_DER_cur + R_DER_cur × 信号值
    constraints += [
        P_dis[:, 0:1, :] - P_ch[:, 0:1, :]
        == cp.reshape(P_DER_cur, (NOFDER, 1, 1))
        + cp.multiply(cp.reshape(R_DER_cur, (NOFDER, 1, 1)), d_s)
    ]

    # 3. VPP投标约束
    constraints += [Bid_P == cp.sum(P_DER, axis=0)]  # 基础功率投标 = 各资源基础功率之和
    constraints += [Bid_R == cp.sum(R_DER, axis=0)]  # 调频功率投标 = 各资源调频功率之和

    # 4. 功率上下限约束 (当前时段和剩余时段)
    constraints += [
        P_dis >= param_std.power_dis_lower_limit[:, cur_slot_idx:, None],
        P_ch >= param_std.power_ch_lower_limit[:, cur_slot_idx:, None],
        P_dis <= param_std.power_dis_upper_limit[:, cur_slot_idx:, None],
        P_ch <= param_std.power_ch_upper_limit[:, cur_slot_idx:, None],
    ]

    # 5. 老化成本计算
    cost = cp.sum(
        cp.multiply(param_std.pr_dis[:, None, None], P_dis)
        + cp.multiply(param_std.pr_ch[:, None, None], P_ch),
        axis=0,
    )
    constraints += [Cost_perf == cost]

    # 6. 能量约束 (软约束，允许松弛)
    constraints += [
        param_std.energy_lower_limit[:, cur_slot_idx:] <= E[:, 1:] + delta_E1[:, 1:],  # 下限+松弛
        E[:, 1:] - delta_E2[:, 1:] <= param_std.energy_upper_limit[:, cur_slot_idx:],  # 上限-松弛
        0 <= delta_E1,  # 松弛变量非负
        0 <= delta_E2,
    ]

    # 7. 响应能力约束 (软约束)
    # 向下调频约束: 能量不能低于下限 (允许松弛delta_E3)
    constraints += [
        E[:, 1:-1]
        - delta_t_req * (param_std.eta_dis @ P_dis[:, 1:, -1])
        >= param_std.energy_lower_limit[:, cur_slot_idx:-1] - delta_E3
    ]
    # 向上调频约束: 能量不能超过上限 (允许松弛delta_E4)
    constraints += [
        E[:, 1:-1]
        - delta_t_req * (param_std.eta_ch @ P_ch[:, 1:, 0])
        <= param_std.energy_upper_limit[:, cur_slot_idx:-1] + delta_E4
    ]
    constraints += [0 <= delta_E3, 0 <= delta_E4]  # 松弛变量非负

    # 8. 能量动态方程 (期望值)
    dist_all = param.hourly_Distribution[cur_slot_idx:, :]  # 当前时段+剩余时段的信号分布
    temp_ch = cp.sum(cp.multiply(P_ch, dist_all[None, :, :]), axis=2)  # 加权充电功率
    temp_dis = cp.sum(cp.multiply(P_dis, dist_all[None, :, :]), axis=2)  # 加权放电功率

    # 剩余时段能量更新
    constraints += [
        E[:, 2:]
        ==  E[:, 1:-1]
        + (param_std.eta_ch @ temp_ch[:, 1:]) * delta_t
        - (param_std.eta_dis @ temp_dis[:, 1:]) * delta_t
    ]

    # 当前时段剩余时间的能量更新
    constraints += [
        E[:, 1]
        == E[:, 0]
        + (param_std.eta_ch @ temp_ch[:, 0]) * delta_t_rest
        - (param_std.eta_dis @ temp_dis[:, 0]) * delta_t_rest
    ]

    # 9. 非调频资源约束
    # param.index_none_reg 已经是0-based索引（在prepare_std.py中计算）
    # none_reg = param.index_none_reg.tolist()
    # 如果有不参与调频的资源，则添加约束（空数组时跳过）
    # if len(none_reg) > 0:
    #     constraints += [R_DER[none_reg, :] == 0]

    # ========== 求解优化问题 ==========
    objective = cp.Maximize(Profit)
    problem = cp.Problem(objective, constraints)
    solver_name = _choose_solver("GUROBI")
    problem.solve(solver=solver_name, verbose=False)

    # ========== 存储优化结果 ==========
    ok = problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    if ok:
        print(f"slot {cur_slot}: rolling bid ok at t_cap {t_cap}")
        # 更新剩余时段的投标
        result["Bid_R_rev"][cur_slot_idx + 1 :] = Bid_R.value
        result["Bid_P_rev"][cur_slot_idx + 1 :] = Bid_P.value
        result["P_DER_rev"][:, cur_slot_idx + 1 :] = P_DER.value
        result["R_DER_rev"][:, cur_slot_idx + 1 :] = R_DER.value
        # result["p_none_reg"] = P_ch.value[none_reg, 0, 0]  # 非调频资源的充电功率
    else:
        print(f"slot {cur_slot}: rolling bid failed at t_cap {t_cap}")
