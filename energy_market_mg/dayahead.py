from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np

from .data_utils import MicrogridCase


def _pick_solver() -> str:
    installed = cp.installed_solvers()
    for name in ("OSQP", "ECOS", "SCS"):
        if name in installed:
            return name
    return "SCS"


def solve_energy_dispatch(
    *,
    price_buy: np.ndarray,
    price_sell: np.ndarray,
    pv_available: np.ndarray,
    load_profile: np.ndarray,
    ev_power_limit: np.ndarray,
    ev_cumulative_demand: np.ndarray,
    es_energy_init: float,
    es_energy_min: float,
    es_energy_max: float,
    es_power_ch_max: float,
    es_power_dis_max: float,
    eta_ch: float,
    eta_dis: float,
    grid_limit: float,
    deg_cost: float,
    curtail_penalty: float,
    ramp_penalty: float,
    shortage_penalty: float,
    delta_t: float,
    prev_grid: float = 0.0,
) -> dict[str, Any]:
    n = len(price_buy)

    grid_import = cp.Variable(n, nonneg=True)
    grid_export = cp.Variable(n, nonneg=True)
    es_charge = cp.Variable(n, nonneg=True)
    es_discharge = cp.Variable(n, nonneg=True)
    ev_charge = cp.Variable(n, nonneg=True)
    pv_used = cp.Variable(n, nonneg=True)
    pv_curt = cp.Variable(n, nonneg=True)
    es_energy = cp.Variable(n + 1)
    ev_shortfall = cp.Variable(n, nonneg=True)

    net_grid = grid_import - grid_export
    cumulative_ev = cp.cumsum(ev_charge) * delta_t

    constraints = [es_energy[0] == es_energy_init]
    constraints += [es_energy[1:] == es_energy[:-1] + eta_ch * es_charge * delta_t - (1.0 / eta_dis) * es_discharge * delta_t]
    constraints += [es_energy >= es_energy_min, es_energy <= es_energy_max]
    constraints += [es_charge <= es_power_ch_max, es_discharge <= es_power_dis_max]
    constraints += [grid_import <= grid_limit, grid_export <= grid_limit]
    constraints += [ev_charge <= ev_power_limit]
    constraints += [pv_used + pv_curt == pv_available]
    constraints += [cumulative_ev + ev_shortfall >= ev_cumulative_demand]
    constraints += [cumulative_ev <= ev_cumulative_demand[-1]]
    constraints += [
        grid_import - grid_export + es_discharge + pv_used
        == load_profile + ev_charge + es_charge
    ]

    if n == 1:
        ramp_expr = cp.reshape(net_grid[0] - prev_grid, (1,), order="F")
    else:
        ramp_expr = cp.hstack(
            [
                cp.reshape(net_grid[0] - prev_grid, (1,), order="F"),
                net_grid[1:] - net_grid[:-1],
            ]
        )
    objective = cp.Minimize(
        price_buy @ grid_import
        - price_sell @ grid_export
        + deg_cost * cp.sum(es_charge + es_discharge)
        + curtail_penalty * cp.sum(pv_curt)
        + shortage_penalty * cp.sum(ev_shortfall)
        + ramp_penalty * cp.sum_squares(ramp_expr)
    )

    problem = cp.Problem(objective, constraints)
    solver_name = _pick_solver()
    problem.solve(solver=solver_name, warm_start=True, verbose=False)

    ok = problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    if not ok:
        raise RuntimeError(f"Energy dispatch failed with status {problem.status}")

    return {
        "grid_import": np.asarray(grid_import.value).flatten(),
        "grid_export": np.asarray(grid_export.value).flatten(),
        "net_grid": np.asarray(net_grid.value).flatten(),
        "es_charge": np.asarray(es_charge.value).flatten(),
        "es_discharge": np.asarray(es_discharge.value).flatten(),
        "es_energy": np.asarray(es_energy.value).flatten(),
        "ev_charge": np.asarray(ev_charge.value).flatten(),
        "ev_shortfall": np.asarray(ev_shortfall.value).flatten(),
        "pv_used": np.asarray(pv_used.value).flatten(),
        "pv_curt": np.asarray(pv_curt.value).flatten(),
        "objective": float(problem.value),
        "status": problem.status,
    }


def solve_day_ahead(case: MicrogridCase) -> dict[str, Any]:
    result = solve_energy_dispatch(
        price_buy=case.energy_price,
        price_sell=case.sell_price,
        pv_available=case.pv_floor_dayahead,
        load_profile=case.load_ceiling_dayahead,
        ev_power_limit=case.ev_power_floor_dayahead,
        ev_cumulative_demand=case.ev_cum_demand_ceiling_dayahead,
        es_energy_init=case.es_energy_init_mwh,
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
    )
    result["pv_available_used_in_model"] = case.pv_floor_dayahead.copy()
    result["load_used_in_model"] = case.load_ceiling_dayahead.copy()
    result["ev_power_used_in_model"] = case.ev_power_floor_dayahead.copy()
    result["ev_cum_demand_used_in_model"] = case.ev_cum_demand_ceiling_dayahead.copy()
    return result

