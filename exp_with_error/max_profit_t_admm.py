from __future__ import annotations

from typing import Any, Dict, List

import cvxpy as cp
import numpy as np

from exp_with_error.max_profit_1_admm import (
    AdmmSettings,
    _compute_tolerances,
    _diag_entries,
    _infer_resource_blocks,
    _load_settings,
    _solve_problem,
)


def _build_rolling_block_constraints(
    row_indices: np.ndarray,
    p_der_future: cp.Variable,
    r_der_future: cp.Variable,
    p_dis: cp.Variable,
    p_ch: cp.Variable,
    energy: cp.Variable,
    slack_low: cp.Variable,
    slack_up: cp.Variable,
    slack_resp_down: cp.Variable,
    slack_resp_up: cp.Variable,
    p_der_cur: np.ndarray,
    r_der_cur: np.ndarray,
    e_cur: np.ndarray,
    param: Any,
    param_std: Any,
    cur_slot_idx: int,
    rest_slots: int,
    nofscen: int,
    delta_t: float,
    delta_t_rest: float,
    delta_t_req: float,
) -> tuple[list[cp.Constraint], cp.Expression, cp.Expression]:
    dist_all = np.asarray(param.hourly_Distribution)[cur_slot_idx:, :]
    d_s = np.asarray(param.d_s).reshape((1, 1, nofscen))

    eta_ch = _diag_entries(np.asarray(param_std.eta_ch), row_indices).reshape(-1, 1)
    eta_dis = _diag_entries(np.asarray(param_std.eta_dis), row_indices).reshape(-1, 1)
    pr_dis = np.asarray(param_std.pr_dis)[row_indices].reshape(-1, 1, 1)
    pr_ch = np.asarray(param_std.pr_ch)[row_indices].reshape(-1, 1, 1)

    energy_lower = np.asarray(param_std.energy_lower_limit)[row_indices, cur_slot_idx:]
    energy_upper = np.asarray(param_std.energy_upper_limit)[row_indices, cur_slot_idx:]
    power_dis_lower = np.asarray(param_std.power_dis_lower_limit)[row_indices, cur_slot_idx:]
    power_dis_upper = np.asarray(param_std.power_dis_upper_limit)[row_indices, cur_slot_idx:]
    power_ch_lower = np.asarray(param_std.power_ch_lower_limit)[row_indices, cur_slot_idx:]
    power_ch_upper = np.asarray(param_std.power_ch_upper_limit)[row_indices, cur_slot_idx:]

    constraints: list[cp.Constraint] = [energy[:, 0] == e_cur]
    constraints += [
        p_dis[:, 1:, :] - p_ch[:, 1:, :]
        == cp.reshape(p_der_future, (len(row_indices), rest_slots, 1), order="F")
        + cp.multiply(cp.reshape(r_der_future, (len(row_indices), rest_slots, 1), order="F"), d_s)
    ]
    constraints += [
        p_dis[:, 0:1, :] - p_ch[:, 0:1, :]
        == cp.reshape(p_der_cur, (len(row_indices), 1, 1), order="F")
        + cp.multiply(cp.reshape(r_der_cur, (len(row_indices), 1, 1), order="F"), d_s)
    ]
    constraints += [
        p_dis >= power_dis_lower[:, :, None],
        p_ch >= power_ch_lower[:, :, None],
        p_dis <= power_dis_upper[:, :, None],
        p_ch <= power_ch_upper[:, :, None],
    ]

    cost_deg = cp.sum(cp.multiply(pr_dis, p_dis) + cp.multiply(pr_ch, p_ch), axis=0)

    constraints += [
        energy_lower <= energy[:, 1:] + slack_low,
        energy[:, 1:] - slack_up <= energy_upper,
        0 <= slack_low,
        0 <= slack_up,
        0 <= slack_resp_down,
        0 <= slack_resp_up,
    ]

    constraints += [
        energy[:, 1:-1] - delta_t_req * cp.multiply(eta_dis, p_dis[:, 1:, -1])
        >= energy_lower[:, :-1] - slack_resp_down
    ]
    constraints += [
        energy[:, 1:-1] - delta_t_req * cp.multiply(eta_ch, p_ch[:, 1:, 0])
        <= energy_upper[:, :-1] + slack_resp_up
    ]

    temp_ch = cp.sum(cp.multiply(p_ch, dist_all[None, :, :]), axis=2)
    temp_dis = cp.sum(cp.multiply(p_dis, dist_all[None, :, :]), axis=2)

    constraints += [
        energy[:, 1]
        == energy[:, 0]
        + cp.multiply(eta_ch, temp_ch[:, 0]) * delta_t_rest
        - cp.multiply(eta_dis, temp_dis[:, 0]) * delta_t_rest
    ]
    constraints += [
        energy[:, 2:]
        == energy[:, 1:-1]
        + cp.multiply(eta_ch, temp_ch[:, 1:]) * delta_t
        - cp.multiply(eta_dis, temp_dis[:, 1:]) * delta_t
    ]

    current_cost = cp.sum(cp.multiply(dist_all[0, :], cost_deg[0, :])) * delta_t_rest
    future_cost = cp.sum(cp.multiply(dist_all[1:, :], cost_deg[1:, :])) * delta_t
    slack_penalty = cp.sum(slack_low) + cp.sum(slack_up) + cp.sum(slack_resp_down) + cp.sum(slack_resp_up)
    return constraints, current_cost + future_cost, slack_penalty


def _solve_rolling_ev_user_subproblem(
    row_index: int,
    y_p: np.ndarray,
    y_r: np.ndarray,
    lambda_p: np.ndarray,
    lambda_r: np.ndarray,
    p_der_cur: float,
    r_der_cur: float,
    e_cur: float,
    ctx: Dict[str, Any],
    settings: AdmmSettings,
    cur_slot_idx: int,
    rest_slots: int,
) -> Dict[str, Any]:
    param = ctx["param"]
    param_std = ctx["param_std"]
    nofscen = ctx["NOFSCEN"]
    delta_t = ctx["delta_t"]
    delta_t_rest = ctx["delta_t_rest"]
    delta_t_req = ctx["delta_t_req"]
    big_m = float(ctx["M"])

    p_der_future = cp.Variable((1, rest_slots))
    r_der_future = cp.Variable((1, rest_slots))
    p_dis = cp.Variable((1, rest_slots + 1, nofscen))
    p_ch = cp.Variable((1, rest_slots + 1, nofscen))
    energy = cp.Variable((1, rest_slots + 2))
    slack_low = cp.Variable((1, rest_slots + 1))
    slack_up = cp.Variable((1, rest_slots + 1))
    slack_resp_down = cp.Variable((1, rest_slots))
    slack_resp_up = cp.Variable((1, rest_slots))

    constraints, degradation_cost, slack_penalty = _build_rolling_block_constraints(
        np.array([row_index], dtype=int),
        p_der_future,
        r_der_future,
        p_dis,
        p_ch,
        energy,
        slack_low,
        slack_up,
        slack_resp_down,
        slack_resp_up,
        np.array([p_der_cur]),
        np.array([r_der_cur]),
        np.array([e_cur]),
        param,
        param_std,
        cur_slot_idx,
        rest_slots,
        nofscen,
        delta_t,
        delta_t_rest,
        delta_t_req,
    )

    objective = cp.Minimize(
        degradation_cost
        + big_m * slack_penalty
        + 0.5 * settings.rho_p * cp.sum_squares(p_der_future[0, :] - y_p + lambda_p)
        + 0.5 * settings.rho_r * cp.sum_squares(r_der_future[0, :] - y_r + lambda_r)
    )
    problem = cp.Problem(objective, constraints)
    _solve_problem(problem, settings.solver_preferences, settings.verbose)

    return {
        "status": problem.status,
        "p": np.asarray(p_der_future.value[0, :]).copy() if p_der_future.value is not None else np.zeros(rest_slots),
        "r": np.asarray(r_der_future.value[0, :]).copy() if r_der_future.value is not None else np.zeros(rest_slots),
        "e": np.asarray(energy.value[0, :]).copy() if energy.value is not None else np.zeros(rest_slots + 2),
    }


def _solve_rolling_master_problem(
    central_indices: np.ndarray,
    avg_a_p: np.ndarray,
    avg_a_r: np.ndarray,
    num_ev_users: int,
    p_der_cur: np.ndarray,
    r_der_cur: np.ndarray,
    e_cur: np.ndarray,
    ctx: Dict[str, Any],
    settings: AdmmSettings,
    cur_slot_idx: int,
    rest_slots: int,
) -> Dict[str, Any]:
    param = ctx["param"]
    param_std = ctx["param_std"]
    nofscen = ctx["NOFSCEN"]
    delta_t = ctx["delta_t"]
    delta_t_rest = ctx["delta_t_rest"]
    delta_t_req = ctx["delta_t_req"]
    big_m = float(ctx["M"])
    num_central = len(central_indices)

    avg_p_ev = cp.Variable(rest_slots)
    avg_r_ev = cp.Variable(rest_slots)

    if num_central > 0:
        p_der_future_c = cp.Variable((num_central, rest_slots))
        r_der_future_c = cp.Variable((num_central, rest_slots))
        p_dis_c = cp.Variable((num_central, rest_slots + 1, nofscen))
        p_ch_c = cp.Variable((num_central, rest_slots + 1, nofscen))
        energy_c = cp.Variable((num_central, rest_slots + 2))
        slack_low = cp.Variable((num_central, rest_slots + 1))
        slack_up = cp.Variable((num_central, rest_slots + 1))
        slack_resp_down = cp.Variable((num_central, rest_slots))
        slack_resp_up = cp.Variable((num_central, rest_slots))
        constraints, degradation_cost_c, slack_penalty_c = _build_rolling_block_constraints(
            central_indices,
            p_der_future_c,
            r_der_future_c,
            p_dis_c,
            p_ch_c,
            energy_c,
            slack_low,
            slack_up,
            slack_resp_down,
            slack_resp_up,
            p_der_cur,
            r_der_cur,
            e_cur,
            param,
            param_std,
            cur_slot_idx,
            rest_slots,
            nofscen,
            delta_t,
            delta_t_rest,
            delta_t_req,
        )
        p_central_sum = cp.sum(p_der_future_c, axis=0)
        r_central_sum = cp.sum(r_der_future_c, axis=0)
    else:
        constraints = []
        degradation_cost_c = 0.0
        slack_penalty_c = 0.0
        p_central_sum = np.zeros(rest_slots)
        r_central_sum = np.zeros(rest_slots)
        p_der_future_c = None
        r_der_future_c = None
        energy_c = None

    total_bid_p = p_central_sum + num_ev_users * avg_p_ev
    total_bid_r = r_central_sum + num_ev_users * avg_r_ev
    price_e = np.asarray(param.price_e)[cur_slot_idx + 1 :]
    price_reg = np.asarray(param.price_reg)[cur_slot_idx + 1 :, :]
    hourly_mileage = np.asarray(param.hourly_Mileage)[cur_slot_idx + 1 :]
    hourly_distribution = np.asarray(param.hourly_Distribution)[cur_slot_idx + 1 :, :]
    reg_mileage = price_reg[:, 1] * hourly_mileage
    reg_energy = hourly_distribution @ np.asarray(param.d_s)

    future_profit = (
        price_e @ total_bid_p
        + price_reg[:, 0] @ total_bid_r * float(param.s_perf)
        + reg_mileage @ total_bid_r * float(param.s_perf)
        + reg_energy @ total_bid_r
    ) * delta_t

    objective = cp.Maximize(
        future_profit
        - degradation_cost_c
        - big_m * slack_penalty_c
        - 0.5 * num_ev_users * settings.rho_p * cp.sum_squares(avg_p_ev - avg_a_p)
        - 0.5 * num_ev_users * settings.rho_r * cp.sum_squares(avg_r_ev - avg_a_r)
    )
    problem = cp.Problem(objective, constraints)
    _solve_problem(problem, settings.solver_preferences, settings.verbose)

    return {
        "status": problem.status,
        "avg_p_ev": np.asarray(avg_p_ev.value).copy() if avg_p_ev.value is not None else np.zeros(rest_slots),
        "avg_r_ev": np.asarray(avg_r_ev.value).copy() if avg_r_ev.value is not None else np.zeros(rest_slots),
        "p_central": np.asarray(p_der_future_c.value).copy() if p_der_future_c is not None and p_der_future_c.value is not None else np.zeros((num_central, rest_slots)),
        "r_central": np.asarray(r_der_future_c.value).copy() if r_der_future_c is not None and r_der_future_c.value is not None else np.zeros((num_central, rest_slots)),
        "e_central": np.asarray(energy_c.value).copy() if energy_c is not None and energy_c.value is not None else np.zeros((num_central, rest_slots + 2)),
        "bid_p": np.asarray(total_bid_p.value).copy() if hasattr(total_bid_p, "value") and total_bid_p.value is not None else np.zeros(rest_slots),
        "bid_r": np.asarray(total_bid_r.value).copy() if hasattr(total_bid_r, "value") and total_bid_r.value is not None else np.zeros(rest_slots),
    }


def max_profit_t_admm(ctx: Dict[str, Any]) -> None:
    """滚动时域的 ADMM 分布式修正版本。"""
    param = ctx["param"]
    result = ctx["result"]
    nofslots = ctx["NOFSLOTS"]
    nofder = ctx["NOFDER"]
    t_cap = int(ctx["t_cap"])
    delta_t_rest = float(ctx["delta_t_rest"])

    cur_slot = int(np.ceil(t_cap / 1800.0))
    cur_slot_idx = cur_slot - 1
    rest_slots = nofslots - cur_slot
    if rest_slots <= 0:
        return

    settings = _load_settings(ctx)
    central_indices, ev_indices = _infer_resource_blocks(param, nofder)
    num_ev_users = len(ev_indices)
    if num_ev_users == 0:
        raise ValueError("ADMM rolling version requires EV user blocks.")

    bid_r_cur = result["Bid_R_rev"][cur_slot_idx]
    bid_p_cur = result["Bid_P_rev"][cur_slot_idx]
    e_cur_full = result["E_cur"]
    p_der_cur_full = result["P_DER_rev"][:, cur_slot_idx]
    r_der_cur_full = result["R_DER_rev"][:, cur_slot_idx]
    result["P_DER_cur"] = p_der_cur_full
    result["R_DER_cur"] = r_der_cur_full

    y_p = result.get("admm_t_y_p")
    y_r = result.get("admm_t_y_r")
    lambda_p = result.get("admm_t_lambda_p")
    lambda_r = result.get("admm_t_lambda_r")
    if y_p is None or y_p.shape != (num_ev_users, rest_slots):
        y_p = np.tile(result["Bid_P_rev"][cur_slot_idx + 1 :] / max(num_ev_users, 1), (num_ev_users, 1))
        y_r = np.tile(result["Bid_R_rev"][cur_slot_idx + 1 :] / max(num_ev_users, 1), (num_ev_users, 1))
        lambda_p = np.zeros((num_ev_users, rest_slots))
        lambda_r = np.zeros((num_ev_users, rest_slots))

    primal_history: List[float] = []
    dual_history: List[float] = []
    eps_pri_history: List[float] = []
    eps_dual_history: List[float] = []

    final_user_steps: List[Dict[str, Any]] = []
    master_step: Dict[str, Any] = {}
    converged = False

    for _ in range(settings.max_iter):
        user_steps: List[Dict[str, Any]] = []
        p_local = np.zeros((num_ev_users, rest_slots))
        r_local = np.zeros((num_ev_users, rest_slots))
        a_p = np.zeros((num_ev_users, rest_slots))
        a_r = np.zeros((num_ev_users, rest_slots))

        for local_idx, row_idx in enumerate(ev_indices):
            step = _solve_rolling_ev_user_subproblem(
                row_index=row_idx,
                y_p=y_p[local_idx],
                y_r=y_r[local_idx],
                lambda_p=lambda_p[local_idx],
                lambda_r=lambda_r[local_idx],
                p_der_cur=float(p_der_cur_full[row_idx]),
                r_der_cur=float(r_der_cur_full[row_idx]),
                e_cur=float(e_cur_full[row_idx]),
                ctx=ctx,
                settings=settings,
                cur_slot_idx=cur_slot_idx,
                rest_slots=rest_slots,
            )
            user_steps.append(step)
            p_local[local_idx] = step["p"]
            r_local[local_idx] = step["r"]
            a_p[local_idx] = step["p"] + lambda_p[local_idx]
            a_r[local_idx] = step["r"] + lambda_r[local_idx]

        avg_a_p = a_p.mean(axis=0)
        avg_a_r = a_r.mean(axis=0)
        master_step = _solve_rolling_master_problem(
            central_indices=central_indices,
            avg_a_p=avg_a_p,
            avg_a_r=avg_a_r,
            num_ev_users=num_ev_users,
            p_der_cur=p_der_cur_full[central_indices],
            r_der_cur=r_der_cur_full[central_indices],
            e_cur=e_cur_full[central_indices],
            ctx=ctx,
            settings=settings,
            cur_slot_idx=cur_slot_idx,
            rest_slots=rest_slots,
        )

        prev_y_p = y_p.copy()
        prev_y_r = y_r.copy()
        delta_p = master_step["avg_p_ev"] - avg_a_p
        delta_r = master_step["avg_r_ev"] - avg_a_r
        y_p = a_p + delta_p[None, :]
        y_r = a_r + delta_r[None, :]
        lambda_p = lambda_p + p_local - y_p
        lambda_r = lambda_r + r_local - y_r

        primal_residual = np.sqrt(np.sum((p_local - y_p) ** 2) + np.sum((r_local - y_r) ** 2))
        dual_residual = np.sqrt(
            settings.rho_p ** 2 * np.sum((y_p - prev_y_p) ** 2)
            + settings.rho_r ** 2 * np.sum((y_r - prev_y_r) ** 2)
        )
        eps_pri, eps_dual = _compute_tolerances(p_local, r_local, y_p, y_r, settings)

        primal_history.append(primal_residual)
        dual_history.append(dual_residual)
        eps_pri_history.append(eps_pri)
        eps_dual_history.append(eps_dual)
        final_user_steps = user_steps

        if primal_residual <= eps_pri and dual_residual <= eps_dual:
            converged = True
            break

    p_der_future_full = np.zeros((nofder, rest_slots))
    r_der_future_full = np.zeros((nofder, rest_slots))
    if len(central_indices) > 0:
        p_der_future_full[central_indices, :] = master_step["p_central"]
        r_der_future_full[central_indices, :] = master_step["r_central"]
    for local_idx, row_idx in enumerate(ev_indices):
        p_der_future_full[row_idx, :] = final_user_steps[local_idx]["p"]
        r_der_future_full[row_idx, :] = final_user_steps[local_idx]["r"]

    ok = converged or (primal_history and np.isfinite(primal_history[-1]) and np.isfinite(dual_history[-1]))
    if ok:
        print(f"slot {cur_slot}: rolling ADMM bid ok at t_cap {t_cap}")
        result["Bid_R_rev"][cur_slot_idx + 1 :] = p_der_future_full.sum(axis=0) * 0 + r_der_future_full.sum(axis=0)
        result["Bid_P_rev"][cur_slot_idx + 1 :] = p_der_future_full.sum(axis=0)
        result["P_DER_rev"][:, cur_slot_idx + 1 :] = p_der_future_full
        result["R_DER_rev"][:, cur_slot_idx + 1 :] = r_der_future_full
        result["revision_times"][cur_slot_idx] += 1
    else:
        print(f"slot {cur_slot}: rolling ADMM bid failed at t_cap {t_cap}")

    result["admm_t_converged"] = converged
    result["admm_t_ok"] = ok
    result["admm_t_iterations"] = len(primal_history)
    result["admm_t_primal_history"] = np.asarray(primal_history)
    result["admm_t_dual_history"] = np.asarray(dual_history)
    result["admm_t_eps_pri_history"] = np.asarray(eps_pri_history)
    result["admm_t_eps_dual_history"] = np.asarray(eps_dual_history)
    result["admm_t_y_p"] = y_p
    result["admm_t_y_r"] = y_r
    result["admm_t_lambda_p"] = lambda_p
    result["admm_t_lambda_r"] = lambda_r
    result["admm_t_ev_avg_p"] = master_step.get("avg_p_ev", np.zeros(rest_slots)).copy()
    result["admm_t_ev_avg_r"] = master_step.get("avg_r_ev", np.zeros(rest_slots)).copy()
    result["admm_t_bid_p_master"] = master_step.get("bid_p", np.zeros(rest_slots)).copy()
    result["admm_t_bid_r_master"] = master_step.get("bid_r", np.zeros(rest_slots)).copy()
    result["admm_t_cur_slot"] = cur_slot
    result["admm_t_delta_t_rest"] = delta_t_rest
    result["admm_t_bid_p_cur"] = float(bid_p_cur)
    result["admm_t_bid_r_cur"] = float(bid_r_cur)
