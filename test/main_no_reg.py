from __future__ import annotations

from typing import Any

import numpy as np

from data_prepare_pv_es_ev import data_prepare_main
from fast_control_implement_no_reg import fast_control_implement
from max_profit_1_no_reg import max_profit_1
from max_profit_t_no_reg import max_profit_t


def _normalize_param(param: Any) -> Any:
    # 简化版：只处理能量价格
    param['price_e'] = np.asarray(param['price_e']).flatten()
    param['s_perf'] = float(param['s_perf'])
    return param


def _normalize_param_std(param_std: Any) -> Any:
    kept_indices = slice(0, 42)  # PV, ES, EV 的索引范围
    param_std['energy_init'] = np.asarray(param_std['energy_init'][kept_indices]).flatten()
    param_std['energy_upper_limit'] = np.asarray(param_std['energy_upper_limit'][kept_indices, :])
    param_std['energy_lower_limit'] = np.asarray(param_std['energy_lower_limit'][kept_indices, :])
    param_std['power_dis_upper_limit'] = np.asarray(param_std['power_dis_upper_limit'][kept_indices, :])
    param_std['power_dis_lower_limit'] = np.asarray(param_std['power_dis_lower_limit'][kept_indices, :])
    param_std['power_ch_upper_limit'] = np.asarray(param_std['power_ch_upper_limit'][kept_indices, :])
    param_std['power_ch_lower_limit'] = np.asarray(param_std['power_ch_lower_limit'][kept_indices, :])

    param_std['eta_ch'] = np.asarray(param_std['eta_ch'][kept_indices, :][:, kept_indices])
    param_std['eta_dis'] = np.asarray(param_std['eta_dis'][kept_indices, :][:, kept_indices])
    param_std['pr_dis'] = np.asarray(param_std['pr_dis'][kept_indices]).flatten()
    param_std['pr_ch'] = np.asarray(param_std['pr_ch'][kept_indices]).flatten()
    param_std['wOmiga'] = np.zeros((42, 24))
    return param_std


def main() -> None:
    day_price = 21

    # 直接生成数据，不加载旧的 test.mat
    print("Generating data using data_prepare_pv_es_ev...")
    data = data_prepare_main(day_price=day_price)

    param = _normalize_param(data["param"])
    param_std = _normalize_param_std(data["param_std"])

    NOFSLOTS = int(np.asarray(data["NOFSLOTS"]).squeeze())
    NOFDER = 42
    delta_t = float(np.asarray(data["delta_t"]).squeeze())
    delta_t_req = float(np.asarray(data.get("delta_t_req", 0.5)).squeeze())

    NOFTCAP_bid = 900   # 15分钟更新一次 (900个2秒)
    NOFTCAP_ctrl = 30   # 30个2秒 (1分钟) 控制一次

    result = {
        "P_alloc": np.zeros((NOFDER, 0)),
        "actualCost": np.zeros(NOFSLOTS),
    }

    ctx = {
        "param": param,
        "param_std": param_std,
        "NOFSLOTS": NOFSLOTS,
        "NOFDER": NOFDER,
        "delta_t": delta_t,
        "delta_t_req": delta_t_req,
        "NOFTCAP_ctrl": NOFTCAP_ctrl,
        "result": result,
    }

    # 初始优化
    max_profit_1(ctx)

    # 滚动优化和实时控制
    for t_cap in range(1, (NOFSLOTS - 1) * 1800 + 1):
        if t_cap % NOFTCAP_bid == 1:
            delta_t_rest = delta_t - ((t_cap - 1) % 1800) / 1800.0
            ctx.update({"t_cap": t_cap, "delta_t_rest": delta_t_rest})
            max_profit_t(ctx)

        if t_cap % NOFTCAP_ctrl == 1:
            ctx.update({"t_cap": t_cap})
            fast_control_implement(ctx)

    # 最后一小时
    for t_cap in range((NOFSLOTS - 1) * 1800 + 1, NOFSLOTS * 1800):
        if t_cap % NOFTCAP_ctrl == 1:
            ctx.update({"t_cap": t_cap})
            fast_control_implement(ctx)

    # 计算总利润
    total_cost = result["actualCost"].sum()
    print("\n" + "=" * 50)
    print("Simulation completed!")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Total profit: ${-total_cost:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
