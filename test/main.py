from __future__ import annotations

from typing import Any

import numpy as np

from fast_control_implement import fast_control_implement
from mat_utils import load_mat, as_1d, as_2d
from max_profit_1 import max_profit_1
from max_profit_t import max_profit_t


def _normalize_param(param: Any) -> Any:
    param.price_e = as_1d(param.price_e)
    param.price_reg = as_2d(param.price_reg)
    param.hourly_Mileage = as_1d(param.hourly_Mileage)
    param.hourly_Distribution = as_2d(param.hourly_Distribution)
    param.d_s = as_1d(param.d_s)
    param.index_none_reg = as_1d(param.index_none_reg).astype(int)
    param.s_perf = float(param.s_perf)
    return param


def _normalize_param_std(param_std: Any, NOFDER: int) -> Any:
    kept_indices = slice(0, NOFDER)  # PV, ES, EV 的索引范围
    param_std.energy_init = as_1d(param_std.energy_init[kept_indices])
    param_std.energy_upper_limit = as_2d(param_std.energy_upper_limit[kept_indices, :])
    param_std.energy_lower_limit = as_2d(param_std.energy_lower_limit[kept_indices, :])
    param_std.power_dis_upper_limit = as_2d(param_std.power_dis_upper_limit[kept_indices, :])
    param_std.power_dis_lower_limit = as_2d(param_std.power_dis_lower_limit[kept_indices, :])
    param_std.power_ch_upper_limit = as_2d(param_std.power_ch_upper_limit[kept_indices, :])
    param_std.power_ch_lower_limit = as_2d(param_std.power_ch_lower_limit[kept_indices, :])

    param_std.eta_ch = as_2d(param_std.eta_ch[kept_indices, :][:, kept_indices])
    param_std.eta_dis = as_2d(param_std.eta_dis[kept_indices, :][:, kept_indices])
    param_std.pr_dis = as_1d(param_std.pr_dis[kept_indices])
    param_std.pr_ch = as_1d(param_std.pr_ch[kept_indices])
    param_std.wOmiga = as_2d(np.zeros((42, 96)))
    return param_std


def main(uncertainty_method: str = "scenario", nofder: int = 42) -> None:
    """主函数：运行虚拟电厂优化控制

    Args:
        uncertainty_method: 不确定性处理方法
            - "scenario": 场景优化（使用所有场景）
            - "CVaR": 条件风险价值优化（控制尾部风险）
            - "FICA": 前向-反向约束近似（只使用极端场景）
        nofder: 资源数量（默认42）
    """
    day_price = 21
    # mat_path = f"data_prepare/param_day_{day_price}.mat"
    mat_path = f"test.mat"
    data = load_mat(mat_path).raw

    param = _normalize_param(data["param"])
    param_std = _normalize_param_std(data["param_std"], nofder)  # 使用传入的nofder参数

    Signal_day = as_1d(data["Signal_day"])
    NOFSLOTS = int(np.asarray(data["NOFSLOTS"]).squeeze())
    NOFDER = nofder  # 使用传入的参数
    NOFSCEN = int(np.asarray(data["NOFSCEN"]).squeeze())
    delta_t = float(np.asarray(data["delta_t"]).squeeze())
    M = float(np.asarray(data.get("M", 1e6)).squeeze())
    delta_t_req = float(np.asarray(data.get("delta_t_req", 0.5)).squeeze())

    NOFTCAP_bid = 900
    NOFTCAP_ctrl = 30

    result = {
        "Bid_R_rev": np.zeros(NOFSLOTS),
        "Bid_P_rev": np.zeros(NOFSLOTS),
        "P_DER_rev": np.zeros((NOFDER, NOFSLOTS)),
        "R_DER_rev": np.zeros((NOFDER, NOFSLOTS)),
        "P_alloc": np.zeros((NOFDER, 0)),
        "actualMil": np.zeros(NOFSLOTS),
        "actualEnergy": np.zeros(NOFSLOTS),
        "actualCost": np.zeros(NOFSLOTS),
        "solve_times": [],  # 记录每次max_profit_t的求解时间
    }

    ctx = {
        "param": param,
        "param_std": param_std,
        "Signal_day": Signal_day,
        "NOFSLOTS": NOFSLOTS,
        "NOFDER": NOFDER,
        "NOFSCEN": NOFSCEN,
        "delta_t": delta_t,
        "delta_t_req": delta_t_req,
        "M": M,
        "NOFTCAP_ctrl": NOFTCAP_ctrl,
        "result": result,
        "uncertainty_method": uncertainty_method,
    }

    max_profit_1(ctx)

    for t_cap in range(1, (NOFSLOTS - 1) * 1800 + 1):
        if t_cap % NOFTCAP_bid == 1:
            delta_t_rest = delta_t - ((t_cap - 1) % 1800) / 1800.0
            ctx.update({"t_cap": t_cap, "delta_t_rest": delta_t_rest})
            solve_time = max_profit_t(ctx, uncertainty_method=uncertainty_method)
        if t_cap % NOFTCAP_ctrl == 1:
            ctx.update({"t_cap": t_cap})
            fast_control_implement(ctx)

    for t_cap in range((NOFSLOTS - 1) * 1800 + 1, NOFSLOTS * 1800):
        if t_cap % NOFTCAP_ctrl == 1:
            ctx.update({"t_cap": t_cap})
            fast_control_implement(ctx)

    result["actualEnegyFee"] = param.price_e * result["actualEnergy"]
    result["actualProfit"] = (
        param.price_reg[:, 0] * result["Bid_R_rev"] * param.s_perf
        + (param.price_reg[:, 1] * result["actualMil"]) * param.s_perf
    )
    result["actualProfit"] = result["actualProfit"] * delta_t

    print("done")
    print(f"profit: {result['actualProfit'].sum():.4f}")
    print(f"avg solve time: {np.mean(result['solve_times']):.4f}s")
    
    return result, param


if __name__ == "__main__":
    # 可选方法: "scenario", "CVaR", "FICA"
    uncertainty_method = "FICA"  # 修改这里来切换方法
    nofder = 50 # 资源数量：42(小规模), 80(中等规模), 121(大规模)

    print(f"使用不确定性处理方法: {uncertainty_method}, NOFDER={nofder}")
    main(uncertainty_method=uncertainty_method, nofder=nofder)
