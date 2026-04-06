from __future__ import annotations

from typing import Any

import numpy as np

from .data_utils import MicrogridCase
from .dayahead import solve_energy_dispatch


def run_intraday_rolling(case: MicrogridCase) -> dict[str, Any]:
    n = case.horizon
    delivered_ev = 0.0
    es_energy = case.es_energy_init_mwh
    prev_grid = 0.0

    records = {
        "net_grid": np.zeros(n),
        "grid_import": np.zeros(n),
        "grid_export": np.zeros(n),
        "es_charge": np.zeros(n),
        "es_discharge": np.zeros(n),
        "es_energy_end": np.zeros(n),
        "ev_charge": np.zeros(n),
        "pv_used": np.zeros(n),
        "pv_curt": np.zeros(n),
        "remaining_ev_shortfall": np.zeros(n),
    }
    rolling_snapshots: list[dict[str, Any]] = []

    for t in range(n):
        future_slice = slice(t, n)
        future_ev_due = np.maximum(case.ev_cum_demand_actual[future_slice] - delivered_ev, 0.0)
        future_ev_due = np.maximum.accumulate(future_ev_due)

        rolling = solve_energy_dispatch(
            price_buy=case.energy_price[future_slice],
            price_sell=case.sell_price[future_slice],
            pv_available=case.pv_actual[future_slice],
            load_profile=case.load_actual[future_slice],
            ev_power_limit=case.ev_power_actual[future_slice],
            ev_cumulative_demand=future_ev_due,
            es_energy_init=es_energy,
            es_energy_min=case.es_energy_min_mwh,
            es_energy_max=case.es_energy_max_mwh,
            es_power_ch_max=case.es_power_ch_max_mw,
            es_power_dis_max=case.es_power_dis_max_mw,
            eta_ch=case.eta_ch,
            eta_dis=case.eta_dis,
            grid_limit=case.grid_limit_mw,
            deg_cost=case.deg_cost_mwh,
            curtail_penalty=case.curtail_penalty_mwh,
            ramp_penalty=case.ramp_penalty,
            shortage_penalty=case.shortage_penalty_mwh,
            delta_t=case.delta_t,
            prev_grid=prev_grid,
        )
        rolling_snapshots.append(rolling)

        records["net_grid"][t] = rolling["net_grid"][0]
        records["grid_import"][t] = rolling["grid_import"][0]
        records["grid_export"][t] = rolling["grid_export"][0]
        records["es_charge"][t] = rolling["es_charge"][0]
        records["es_discharge"][t] = rolling["es_discharge"][0]
        records["es_energy_end"][t] = rolling["es_energy"][1]
        records["ev_charge"][t] = rolling["ev_charge"][0]
        records["pv_used"][t] = rolling["pv_used"][0]
        records["pv_curt"][t] = rolling["pv_curt"][0]
        records["remaining_ev_shortfall"][t] = rolling["ev_shortfall"][-1]

        delivered_ev += rolling["ev_charge"][0] * case.delta_t
        es_energy = rolling["es_energy"][1]
        prev_grid = rolling["net_grid"][0]

    records["total_objective_proxy"] = float(
        case.energy_price @ records["grid_import"]
        - case.sell_price @ records["grid_export"]
        + case.deg_cost_mwh * np.sum(records["es_charge"] + records["es_discharge"])
        + case.curtail_penalty_mwh * np.sum(records["pv_curt"])
    )
    records["rolling_snapshots"] = rolling_snapshots
    return records

