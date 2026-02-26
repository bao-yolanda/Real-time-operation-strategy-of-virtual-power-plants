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
    """日前初始投标优化：在仿真开始时优化全天的投标策略

    使用CVXPY求解混合整数规划问题，优化VPP的基础功率和调频功率投标
    """
    # ========== 输入参数 ==========
    param = ctx["param"]         # 市场参数 (价格、里程、分布等)
    param_std = ctx["param_std"] # 资源标准化参数 (55个资源的约束)
    result = ctx["result"]       # 结果字典

    # 控制参数
    NOFSLOTS = ctx["NOFSLOTS"]           # 时间段数 (96, 每15分钟一段)
    NOFDER = ctx["NOFDER"]               # 资源数量 (55)
    NOFSCEN = ctx["NOFSCEN"]             # 场景数 (调频信号离散化后的场景数量)
    delta_t = ctx["delta_t"]             # 时间步长 (0.25小时 = 15分钟)
    delta_t_req = ctx["delta_t_req"]     # 响应时间间隔 (0.5小时)

    # ========== 决策变量 ==========
    Bid_P = cp.Variable(NOFSLOTS)         # VPP基础功率投标 (96×1, MW)
    Bid_R = cp.Variable(NOFSLOTS)         # VPP调频功率投标 (96×1, MW)
    R_DER = cp.Variable((NOFDER, NOFSLOTS))  # 每个资源的调频功率分配 (55×96, MW)
    P_DER = cp.Variable((NOFDER, NOFSLOTS))  # 每个资源的基础功率分配 (55×96, MW)

    # 场景相关变量 (考虑调频信号的不确定性)
    P_dis = cp.Variable((NOFDER, NOFSLOTS, NOFSCEN))  # 放电功率 (55×96×场景数, MW)
    P_ch = cp.Variable((NOFDER, NOFSLOTS, NOFSCEN))    # 充电功率 (55×96×场景数, MW)
    E = cp.Variable((NOFDER, NOFSLOTS + 1))            # 能量状态 (55×97, MWh或等效)
    Cost_deg = cp.Variable((NOFSLOTS, NOFSCEN))        # 老化成本 (96×场景数, $)

    # ========== 市场价格参数 ==========
    price_e = param.price_e                      # 实时能量价格 (96×1, $/MWh)
    price_reg = param.price_reg                  # 调频市场价格 (96×2, 第1列=容量价格, 第2列=里程价格)
    hourly_mileage = param.hourly_Mileage       # 每小时预期里程 (24×1)
    hourly_distribution = param.hourly_Distribution  # 调频信号概率分布 (场景数×24)

    # ========== 收益计算 ==========
    # 1. 里程收益 = 里程价格 × 预期里程 × 调频投标量
    reg_mileage = price_reg[:, 1] * hourly_mileage

    # 2. 调频能量收益 = 信号期望值 × 能量价格
    reg_energy = cp.multiply(hourly_distribution @ param.d_s, price_e)

    # 3. 总利润 = 能量收益 + 容量收益 + 里程收益 - 老化成本
    Profit = (
        price_e @ Bid_P                      # 能量收益: 基础功率 × 能量价格
        + price_reg[:, 0] @ Bid_R * param.s_perf   # 容量收益: 调频功率 × 容量价格 × 性能得分
        + reg_mileage @ Bid_R * param.s_perf      # 里程收益: 调频功率 × 里程期望 × 性能得分
        + reg_energy @ Bid_R                   # 调频能量收益
        - cp.sum(cp.multiply(hourly_distribution, Cost_deg))  # 老化成本
    )
    Profit = Profit * delta_t  # 转换为总收益

    # ========== 约束条件 ==========
    constraints = []

    # 1. 初始能量约束
    constraints += [E[:, 0] == param_std.energy_init]

    # 2. 功量平衡约束: P_dis - P_ch = P_DER + R_DER × 信号值
    d_s = param.d_s.reshape((1, 1, NOFSCEN))  # 场景信号值 (1×1×场景数)
    constraints += [
        P_dis - P_ch                                   # 净放电功率
        == cp.reshape(P_DER, (NOFDER, NOFSLOTS, 1))    # 基础功率 (55×96×1)
        + cp.multiply(cp.reshape(R_DER, (NOFDER, NOFSLOTS, 1)), d_s)  # 调频响应 (55×96×场景数)
    ]

    # 3. VPP投标约束: VPP总投标 = 各资源分配之和
    constraints += [Bid_P == cp.sum(P_DER, axis=0)]   # 基础功率投标 = 各资源基础功率之和
    constraints += [Bid_R == cp.sum(R_DER, axis=0)]   # 调频功率投标 = 各资源调频功率之和

    # 4. 功率上下限约束
    constraints += [
        P_dis >= param_std.power_dis_lower_limit[:, :, None],  # 放电功率下限
        P_ch >= param_std.power_ch_lower_limit[:, :, None],    # 充电功率下限
        P_dis <= param_std.power_dis_upper_limit[:, :, None],  # 放电功率上限
        P_ch <= param_std.power_ch_upper_limit[:, :, None],    # 充电功率上限
    ]

    # 5. 老化成本计算: cost = pr_dis × P_dis + pr_ch × P_ch
    cost = cp.sum(
        cp.multiply(param_std.pr_dis[:, None, None], P_dis)  # ES/EV的放电老化成本
        + cp.multiply(param_std.pr_ch[:, None, None], P_ch),  # ES/EV的充电老化成本
        axis=0,
    )
    constraints += [Cost_deg == cost]

    # 6. 能量约束 (当前时段)
    constraints += [param_std.energy_lower_limit <= E[:, 1:]]  # 能量下限
    constraints += [E[:, 1:] <= param_std.energy_upper_limit]  # 能量上限

    # 7. 响应能力约束: 确保在delta_t_req时间内有足够能量响应调频信号
    lower_shift = np.concatenate(
        [param_std.energy_lower_limit[:, :1], param_std.energy_lower_limit[:, :-1]],
        axis=1,
    )
    theta_factor = 1.0 - delta_t_req * (1.0 - param_std.theta)  # 衰减因子

    # 7a. 向下调频约束: 能量不能低于下限
    constraints += [
        cp.multiply(theta_factor[:, None], E[:, :-1])           # 衰减后的能量
        - delta_t_req * (param_std.eta_dis @ P_dis[:, :, -1])  # 最大放电消耗
        + delta_t_req * param_std.wOmiga                       # 外部影响
        >= lower_shift                                          # 能量下限
    ]

    # 7b. 向上调频约束: 能量不能超过上限
    constraints += [
        cp.multiply(theta_factor[:, None], E[:, :-1])           # 衰减后的能量
        - delta_t_req * (param_std.eta_ch @ P_ch[:, :, 0])    # 最大充电输入
        + delta_t_req * param_std.wOmiga                       # 外部影响
        <= param_std.energy_upper_limit                        # 能量上限
    ]

    # 8. 能量动态方程 (期望值)
    dist = hourly_distribution
    temp_ch = cp.sum(cp.multiply(P_ch, dist[None, :, :]), axis=2)  # 加权充电功率 (55×96)
    temp_dis = cp.sum(cp.multiply(P_dis, dist[None, :, :]), axis=2) # 加权放电功率 (55×96)

    constraints += [
        E[:, 1:]                                               # 下一时刻能量
        == cp.multiply(param_std.theta[:, None], E[:, :-1])  # 衰减项
        + (param_std.eta_ch @ temp_ch) * delta_t              # 充电输入项
        - (param_std.eta_dis @ temp_dis) * delta_t             # 放电输出项
        + param_std.wOmiga * delta_t                           # 外部影响
    ]

    # 9. 非调频资源约束: 某些资源(如IPP的某些环节)不参与调频
    none_reg = (param.index_none_reg.astype(int) - 1).tolist()  # 不参与调频的资源索引
    constraints += [R_DER[none_reg, :] == 0]  # 这些资源的调频功率必须为0

    # ========== 求解优化问题 ==========
    objective = cp.Maximize(Profit)
    problem = cp.Problem(objective, constraints)
    solver_name = _choose_solver("GUROBI")  # 优先使用Gurobi求解器
    problem.solve(solver=solver_name, verbose=False)

    # ========== 检查求解状态 ==========
    ok = problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    if ok:
        print("slot 1: bidding ok")
    else:
        print("slot 1: bidding failed")

    # ========== 存储初始投标结果 (全天最优解) ==========
    result["Bid_R_init"] = Bid_R.value    # 初始调频投标 (96×1, MW)
    result["Bid_P_init"] = Bid_P.value    # 初始基础功率投标 (96×1, MW)
    result["E_init"] = E.value              # 初始能量状态 (55×97)

    # ========== 存储当前时段状态 (用于实时控制) ==========
    result["Bid_R_cur"] = Bid_R.value[0]   # 当前时段调频投标 (标量, MW)
    result["Bid_P_cur"] = Bid_P.value[0]   # 当前时段基础功率投标 (标量, MW)
    result["E_cur"] = E.value[:, 0]        # 当前能量状态 (55×1)
    result["P_DER_cur"] = P_DER.value[:, 0]  # 当前各资源基础功率 (55×1, MW)
    result["R_DER_cur"] = R_DER.value[:, 0]  # 当前各资源调频功率 (55×1, MW)

    # ========== 存储投标历史记录 ==========
    result["Bid_R_rev"] = Bid_R.value.copy()  # 调频投标历史 (96×1)
    result["Bid_P_rev"] = Bid_P.value.copy()  # 基础功率投标历史 (96×1)
    result["P_DER_rev"] = P_DER.value.copy()  # 各资源基础功率历史 (55×96)
    result["R_DER_rev"] = R_DER.value.copy()  # 各资源调频功率历史 (55×96)
    result["E_rev"] = result["E_cur"].reshape(-1, 1)  # 能量历史 (55×1, 初始状态)
