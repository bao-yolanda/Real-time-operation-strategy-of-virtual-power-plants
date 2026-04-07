from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Dict, List, Sequence

import cvxpy as cp
import numpy as np


def _choose_solver(preferred: Sequence[str] | str) -> str:
    installed = cp.installed_solvers()
    preferred_list = [preferred] if isinstance(preferred, str) else list(preferred)
    for name in preferred_list:
        if name in installed:
            return name
    for fallback in ["OSQP", "ECOS", "SCS", "GUROBI"]:
        if fallback in installed:
            return fallback
    return "SCS"


def _solve_problem(problem: cp.Problem, preferred: Sequence[str] | str, verbose: bool) -> None:
    tried: List[str] = []
    solver_order = [preferred] if isinstance(preferred, str) else list(preferred)
    solver_order.extend(["OSQP", "ECOS", "SCS", "GUROBI"])
    for solver_name in solver_order:
        if solver_name in tried or solver_name not in cp.installed_solvers():
            continue
        tried.append(solver_name)
        try:
            problem.solve(solver=solver_name, verbose=verbose, warm_start=True)
            return
        except Exception:
            continue
    problem.solve(verbose=verbose, warm_start=True)


@dataclass
class AdmmSettings:
    max_iter: int = 40
    rho_p: float = 5.0
    rho_r: float = 5.0
    abs_tol: float = 1e-3
    rel_tol: float = 1e-3
    verbose: bool = False
    solver_preferences: tuple[str, ...] = ("OSQP", "ECOS", "SCS", "GUROBI")
    parallel_user_solve: bool = True
    user_solver_workers: int = 0
    progress_every: int = 5


def _load_settings(ctx: Dict[str, Any]) -> AdmmSettings:
    raw = ctx.get("admm_settings") or ctx.get("admm_params") or {}
    return AdmmSettings(
        max_iter=int(raw.get("max_iter", 40)),
        rho_p=float(raw.get("rho_p", raw.get("rho", 5.0))),
        rho_r=float(raw.get("rho_r", raw.get("rho", 5.0))),
        abs_tol=float(raw.get("abs_tol", 1e-3)),
        rel_tol=float(raw.get("rel_tol", 1e-3)),
        verbose=bool(raw.get("verbose", False)),
        solver_preferences=tuple(raw.get("solver_preferences", ("OSQP", "ECOS", "SCS", "GUROBI"))),
        parallel_user_solve=bool(raw.get("parallel_user_solve", True)),
        user_solver_workers=int(raw.get("user_solver_workers", 0)),
        progress_every=max(1, int(raw.get("progress_every", 5))),
    )


def _resolve_user_solver_workers(settings: AdmmSettings, num_jobs: int) -> int:
    if num_jobs <= 1 or not settings.parallel_user_solve:
        return 1
    if settings.user_solver_workers > 0:
        return max(1, min(num_jobs, settings.user_solver_workers))
    return max(1, min(num_jobs, os.cpu_count() or 1))


def _run_user_solver_jobs(
    solver_fn: Callable[..., Dict[str, Any]],
    jobs: Sequence[tuple[Any, ...]],
    settings: AdmmSettings,
) -> List[Dict[str, Any]]:
    if not jobs:
        return []

    workers = _resolve_user_solver_workers(settings, len(jobs))
    if workers == 1:
        return [solver_fn(*job) for job in jobs]

    results: List[Dict[str, Any] | None] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(solver_fn, *job): idx
            for idx, job in enumerate(jobs)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
    return [result for result in results if result is not None]


def _infer_resource_blocks(param: Any, nofder: int) -> tuple[np.ndarray, np.ndarray]:
    inferred_ev = getattr(param, "NOFEV", None)
    if inferred_ev is None:
        inferred_ev = max(nofder - 2, 0)
    num_ev = int(max(0, min(inferred_ev, nofder)))
    num_central = max(nofder - num_ev, 0)
    if num_central == 0 and nofder >= 2:
        num_central = 2
        num_ev = nofder - num_central
    central_indices = np.arange(num_central, dtype=int)
    ev_indices = np.arange(num_central, num_central + num_ev, dtype=int)
    return central_indices, ev_indices


def _diag_entries(matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.diag(np.asarray(matrix))[indices]


def _build_block_constraints(
    row_indices: np.ndarray,
    p_der: cp.Variable,
    r_der: cp.Variable,
    p_dis: cp.Variable,
    p_ch: cp.Variable,
    energy: cp.Variable,
    param: Any,
    param_std: Any,
    nofslots: int,
    nofscen: int,
    delta_t: float,
    delta_t_req: float,
) -> tuple[list[cp.Constraint], cp.Expression]:
    dist = np.asarray(param.hourly_Distribution)
    d_s = np.asarray(param.d_s).reshape((1, 1, nofscen))

    eta_ch = _diag_entries(np.asarray(param_std.eta_ch), row_indices).reshape(-1, 1)
    eta_dis = _diag_entries(np.asarray(param_std.eta_dis), row_indices).reshape(-1, 1)
    pr_dis = np.asarray(param_std.pr_dis)[row_indices].reshape(-1, 1, 1)
    pr_ch = np.asarray(param_std.pr_ch)[row_indices].reshape(-1, 1, 1)

    energy_init = np.asarray(param_std.energy_init)[row_indices]
    energy_lower = np.asarray(param_std.energy_lower_limit)[row_indices]
    energy_upper = np.asarray(param_std.energy_upper_limit)[row_indices]
    power_dis_lower = np.asarray(param_std.power_dis_lower_limit)[row_indices]
    power_dis_upper = np.asarray(param_std.power_dis_upper_limit)[row_indices]
    power_ch_lower = np.asarray(param_std.power_ch_lower_limit)[row_indices]
    power_ch_upper = np.asarray(param_std.power_ch_upper_limit)[row_indices]
    constraints: list[cp.Constraint] = [energy[:, 0] == energy_init]
    constraints += [
        p_dis - p_ch
        == cp.reshape(p_der, (len(row_indices), nofslots, 1), order="F")
        + cp.multiply(cp.reshape(r_der, (len(row_indices), nofslots, 1), order="F"), d_s)
    ]
    constraints += [
        p_dis >= power_dis_lower[:, :, None],
        p_ch >= power_ch_lower[:, :, None],
        p_dis <= power_dis_upper[:, :, None],
        p_ch <= power_ch_upper[:, :, None],
    ]

    cost_deg = cp.sum(cp.multiply(pr_dis, p_dis) + cp.multiply(pr_ch, p_ch), axis=0)

    constraints += [energy_lower <= energy[:, 1:]]
    constraints += [energy[:, 1:] <= energy_upper]

    lower_shift = np.concatenate([energy_lower[:, :1], energy_lower[:, :-1]], axis=1)
    constraints += [
        energy[:, :-1]
        - delta_t_req * cp.multiply(eta_dis, p_dis[:, :, -1])
        >= lower_shift
    ]
    constraints += [
        energy[:, :-1]
        - delta_t_req * cp.multiply(eta_ch, p_ch[:, :, 0])
        <= energy_upper
    ]

    temp_ch = cp.sum(cp.multiply(p_ch, dist[None, :, :]), axis=2)
    temp_dis = cp.sum(cp.multiply(p_dis, dist[None, :, :]), axis=2)
    constraints += [
        energy[:, 1:]
        == energy[:, :-1]
        + cp.multiply(eta_ch, temp_ch) * delta_t
        - cp.multiply(eta_dis, temp_dis) * delta_t
    ]
    return constraints, cost_deg


def _expected_deg_cost_per_slot(
    row_indices: np.ndarray,
    p_dis: np.ndarray | None,
    p_ch: np.ndarray | None,
    param: Any,
    param_std: Any,
    delta_t: float,
) -> np.ndarray:
    dist = np.asarray(param.hourly_Distribution)
    if len(row_indices) == 0 or p_dis is None or p_ch is None:
        return np.zeros(dist.shape[0])

    p_dis_array = np.asarray(p_dis)
    p_ch_array = np.asarray(p_ch)
    if p_dis_array.ndim == 2:
        p_dis_array = p_dis_array[None, :, :]
    if p_ch_array.ndim == 2:
        p_ch_array = p_ch_array[None, :, :]

    pr_dis = np.asarray(param_std.pr_dis)[row_indices].reshape(-1, 1, 1)
    pr_ch = np.asarray(param_std.pr_ch)[row_indices].reshape(-1, 1, 1)
    cost_deg = np.sum(pr_dis * p_dis_array + pr_ch * p_ch_array, axis=0)
    return delta_t * np.sum(dist * cost_deg, axis=1)


def _solve_ev_fleet_subproblem(
    row_indices: np.ndarray,
    z_p: np.ndarray,
    z_r: np.ndarray,
    lambda_p: np.ndarray,
    lambda_r: np.ndarray,
    ctx: Dict[str, Any],
    settings: AdmmSettings,
) -> Dict[str, Any]:
    param = ctx["param"]
    param_std = ctx["param_std"]
    nofslots = ctx["NOFSLOTS"]
    nofscen = ctx["NOFSCEN"]
    delta_t = ctx["delta_t"]
    delta_t_req = ctx["delta_t_req"]
    num_ev = len(row_indices)

    p_der = cp.Variable((num_ev, nofslots))
    r_der = cp.Variable((num_ev, nofslots))
    p_dis = cp.Variable((num_ev, nofslots, nofscen))
    p_ch = cp.Variable((num_ev, nofslots, nofscen))
    energy = cp.Variable((num_ev, nofslots + 1))

    constraints, cost_deg = _build_block_constraints(
        row_indices,
        p_der,
        r_der,
        p_dis,
        p_ch,
        energy,
        param,
        param_std,
        nofslots,
        nofscen,
        delta_t,
        delta_t_req,
    )
    p_sum = cp.sum(p_der, axis=0)
    r_sum = cp.sum(r_der, axis=0)

    objective = cp.Minimize(
        delta_t * cp.sum(cp.multiply(np.asarray(param.hourly_Distribution), cost_deg))
        + 0.5 * settings.rho_p * cp.sum_squares(p_sum - z_p + lambda_p)
        + 0.5 * settings.rho_r * cp.sum_squares(r_sum - z_r + lambda_r)
    )
    problem = cp.Problem(objective, constraints)
    solve_start = perf_counter()
    _solve_problem(problem, settings.solver_preferences, settings.verbose)
    solve_time = perf_counter() - solve_start

    return {
        "status": problem.status,
        "solve_time": solve_time,
        "p": np.asarray(p_der.value).copy() if p_der.value is not None else np.zeros((num_ev, nofslots)),
        "r": np.asarray(r_der.value).copy() if r_der.value is not None else np.zeros((num_ev, nofslots)),
        "e": np.asarray(energy.value).copy() if energy.value is not None else np.zeros((num_ev, nofslots + 1)),
        "p_dis": np.asarray(p_dis.value).copy() if p_dis.value is not None else np.zeros((num_ev, nofslots, nofscen)),
        "p_ch": np.asarray(p_ch.value).copy() if p_ch.value is not None else np.zeros((num_ev, nofslots, nofscen)),
        "p_sum": np.asarray(p_sum.value).copy() if p_sum.value is not None else np.zeros(nofslots),
        "r_sum": np.asarray(r_sum.value).copy() if r_sum.value is not None else np.zeros(nofslots),
    }


def _solve_master_problem(
    central_indices: np.ndarray,
    agg_a_p: np.ndarray,
    agg_a_r: np.ndarray,
    ctx: Dict[str, Any],
    settings: AdmmSettings,
) -> Dict[str, Any]:
    param = ctx["param"]
    param_std = ctx["param_std"]
    nofslots = ctx["NOFSLOTS"]
    nofscen = ctx["NOFSCEN"]
    delta_t = ctx["delta_t"]
    delta_t_req = ctx["delta_t_req"]
    num_central = len(central_indices)

    z_p = cp.Variable(nofslots)
    z_r = cp.Variable(nofslots)

    if num_central > 0:
        p_der_c = cp.Variable((num_central, nofslots))
        r_der_c = cp.Variable((num_central, nofslots))
        p_dis_c = cp.Variable((num_central, nofslots, nofscen))
        p_ch_c = cp.Variable((num_central, nofslots, nofscen))
        energy_c = cp.Variable((num_central, nofslots + 1))
        constraints, cost_deg_c = _build_block_constraints(
            central_indices,
            p_der_c,
            r_der_c,
            p_dis_c,
            p_ch_c,
            energy_c,
            param,
            param_std,
            nofslots,
            nofscen,
            delta_t,
            delta_t_req,
        )
        p_central_sum = cp.sum(p_der_c, axis=0)
        r_central_sum = cp.sum(r_der_c, axis=0)
    else:
        constraints = []
        cost_deg_c = 0.0
        p_central_sum = np.zeros(nofslots)
        r_central_sum = np.zeros(nofslots)
        p_der_c = None
        r_der_c = None
        p_dis_c = None
        p_ch_c = None
        energy_c = None

    total_bid_p = p_central_sum + z_p
    total_bid_r = r_central_sum + z_r
    reg_mileage = np.asarray(param.price_reg)[:, 1] * np.asarray(param.hourly_Mileage)
    reg_energy = np.asarray(param.hourly_Distribution) @ np.asarray(param.d_s)
    profit = (
        np.asarray(param.price_e) @ total_bid_p
        + np.asarray(param.price_reg)[:, 0] @ total_bid_r * float(param.s_perf)
        + reg_mileage @ total_bid_r * float(param.s_perf)
        + reg_energy @ total_bid_r
        - cp.sum(cp.multiply(np.asarray(param.hourly_Distribution), cost_deg_c))
    ) * delta_t

    objective = cp.Maximize(
        profit
        - 0.5 * settings.rho_p * cp.sum_squares(z_p - agg_a_p)
        - 0.5 * settings.rho_r * cp.sum_squares(z_r - agg_a_r)
    )
    problem = cp.Problem(objective, constraints)
    solve_start = perf_counter()
    _solve_problem(problem, settings.solver_preferences, settings.verbose)
    solve_time = perf_counter() - solve_start

    return {
        "status": problem.status,
        "solve_time": solve_time,
        "z_p": np.asarray(z_p.value).copy() if z_p.value is not None else np.zeros(nofslots),
        "z_r": np.asarray(z_r.value).copy() if z_r.value is not None else np.zeros(nofslots),
        "p_central": np.asarray(p_der_c.value).copy() if p_der_c is not None and p_der_c.value is not None else np.zeros((num_central, nofslots)),
        "r_central": np.asarray(r_der_c.value).copy() if r_der_c is not None and r_der_c.value is not None else np.zeros((num_central, nofslots)),
        "p_dis_c": np.asarray(p_dis_c.value).copy() if p_dis_c is not None and p_dis_c.value is not None else np.zeros((num_central, nofslots, nofscen)),
        "p_ch_c": np.asarray(p_ch_c.value).copy() if p_ch_c is not None and p_ch_c.value is not None else np.zeros((num_central, nofslots, nofscen)),
        "e_central": np.asarray(energy_c.value).copy() if energy_c is not None and energy_c.value is not None else np.zeros((num_central, nofslots + 1)),
        "bid_p": np.asarray(total_bid_p.value).copy() if hasattr(total_bid_p, "value") and total_bid_p.value is not None else np.zeros(nofslots),
        "bid_r": np.asarray(total_bid_r.value).copy() if hasattr(total_bid_r, "value") and total_bid_r.value is not None else np.zeros(nofslots),
    }


def _compute_tolerances(
    p_local: np.ndarray,
    r_local: np.ndarray,
    y_p: np.ndarray,
    y_r: np.ndarray,
    settings: AdmmSettings,
) -> tuple[float, float]:
    dim = p_local.size + r_local.size
    eps_pri = np.sqrt(dim) * settings.abs_tol + settings.rel_tol * max(
        np.linalg.norm(p_local.ravel()),
        np.linalg.norm(y_p.ravel()),
        np.linalg.norm(r_local.ravel()),
        np.linalg.norm(y_r.ravel()),
    )
    eps_dual = np.sqrt(dim) * settings.abs_tol + settings.rel_tol * max(
        settings.rho_p * np.linalg.norm(y_p.ravel()),
        settings.rho_r * np.linalg.norm(y_r.ravel()),
    )
    return eps_pri, eps_dual


def max_profit_1_admm(ctx: Dict[str, Any]) -> None:
    """日前初始投标优化的 ADMM 版本。

    采用“两块 ADMM”的结构：
    - 主块：PV/ES 与市场投标；
    - EV块：保留所有 EV 行约束的整块矩阵子问题；
    - 两块围绕 EV 聚合后的总 p/r 做一致性迭代。
    """
    param = ctx["param"]
    param_std = ctx["param_std"]
    result = ctx["result"]
    nofslots = ctx["NOFSLOTS"]
    nofder = ctx["NOFDER"]

    settings = _load_settings(ctx)
    central_indices, ev_indices = _infer_resource_blocks(param, nofder)
    num_ev_rows = len(ev_indices)

    if num_ev_rows == 0:
        raise ValueError("ADMM 版本需要至少一个 EV 子块；当前 NOFEV=0。")

    z_p = np.zeros(nofslots)
    z_r = np.zeros(nofslots)
    lambda_p = np.zeros(nofslots)
    lambda_r = np.zeros(nofslots)

    primal_history: List[float] = []
    dual_history: List[float] = []
    eps_pri_history: List[float] = []
    eps_dual_history: List[float] = []
    user_solve_time_history: List[float] = []
    system_solve_time_history: List[float] = []
    user_solve_time_by_user = np.zeros(1)
    user_solve_counts_by_user = np.zeros(1)

    final_user_step: Dict[str, Any] = {}
    master_step: Dict[str, Any] = {}
    converged = False

    print(
        f"slot 1: start ADMM bidding, max_iter={settings.max_iter}, "
        f"EV_rows={num_ev_rows}, central_rows={len(central_indices)}",
        flush=True,
    )
    for iter_idx in range(1, settings.max_iter + 1):
        local_step = _solve_ev_fleet_subproblem(
            ev_indices,
            z_p,
            z_r,
            lambda_p,
            lambda_r,
            ctx,
            settings,
        )
        user_solve_time_history.append(float(local_step.get("solve_time", 0.0)))
        user_solve_time_by_user[0] += float(local_step.get("solve_time", 0.0))
        user_solve_counts_by_user[0] += 1.0

        p_local = local_step["p_sum"]
        r_local = local_step["r_sum"]
        a_p = p_local + lambda_p
        a_r = r_local + lambda_r

        master_step = _solve_master_problem(
            central_indices,
            a_p,
            a_r,
            ctx,
            settings,
        )

        prev_z_p = z_p.copy()
        prev_z_r = z_r.copy()
        z_p = master_step["z_p"]
        z_r = master_step["z_r"]
        lambda_p = lambda_p + p_local - z_p
        lambda_r = lambda_r + r_local - z_r
        system_solve_time_history.append(float(master_step.get("solve_time", 0.0)))

        primal_residual = np.sqrt(np.sum((p_local - z_p) ** 2) + np.sum((r_local - z_r) ** 2))
        dual_residual = np.sqrt(
            settings.rho_p ** 2 * np.sum((z_p - prev_z_p) ** 2)
            + settings.rho_r ** 2 * np.sum((z_r - prev_z_r) ** 2)
        )
        eps_pri, eps_dual = _compute_tolerances(p_local, r_local, z_p, z_r, settings)

        primal_history.append(primal_residual)
        dual_history.append(dual_residual)
        eps_pri_history.append(eps_pri)
        eps_dual_history.append(eps_dual)

        should_report = (
            iter_idx == 1
            or iter_idx == settings.max_iter
            or iter_idx % settings.progress_every == 0
            or primal_residual <= eps_pri
            or dual_residual <= eps_dual
        )
        if should_report:
            print(
                "slot 1: ADMM iter "
                f"{iter_idx}/{settings.max_iter}, "
                f"primal={primal_residual:.4e}, dual={dual_residual:.4e}, "
                f"eps_pri={eps_pri:.4e}, eps_dual={eps_dual:.4e}",
                flush=True,
            )

        final_user_step = local_step
        if primal_residual <= eps_pri and dual_residual <= eps_dual:
            converged = True
            break

    p_der_full = np.zeros((nofder, nofslots))
    r_der_full = np.zeros((nofder, nofslots))
    e_full = np.zeros((nofder, nofslots + 1))

    if len(central_indices) > 0:
        p_der_full[central_indices, :] = master_step["p_central"]
        r_der_full[central_indices, :] = master_step["r_central"]
        e_full[central_indices, :] = master_step["e_central"]

    p_der_full[ev_indices, :] = final_user_step.get("p", np.zeros((num_ev_rows, nofslots)))
    r_der_full[ev_indices, :] = final_user_step.get("r", np.zeros((num_ev_rows, nofslots)))
    e_full[ev_indices, :] = final_user_step.get("e", np.zeros((num_ev_rows, nofslots + 1)))

    deg_day = np.zeros(nofslots)
    if len(central_indices) > 0:
        deg_day += _expected_deg_cost_per_slot(
            central_indices,
            master_step.get("p_dis_c"),
            master_step.get("p_ch_c"),
            param,
            param_std,
            ctx["delta_t"],
        )

    deg_day += _expected_deg_cost_per_slot(
        ev_indices,
        final_user_step.get("p_dis"),
        final_user_step.get("p_ch"),
        param,
        param_std,
        ctx["delta_t"],
    )

    bid_p = p_der_full.sum(axis=0)
    bid_r = r_der_full.sum(axis=0)
    ok = converged or (primal_history and dual_history and np.isfinite(primal_history[-1]) and np.isfinite(dual_history[-1]))
    if converged:
        print("slot 1: ADMM bidding ok")
    else:
        print("slot 1: ADMM bidding reached iteration limit")

    result["Bid_R_init"] = bid_r
    result["Bid_P_init"] = bid_p
    result["E_init"] = e_full

    result["Bid_R_cur"] = float(bid_r[0]) if bid_r.size else 0.0
    result["Bid_P_cur"] = float(bid_p[0]) if bid_p.size else 0.0
    result["E_cur"] = e_full[:, 0] if e_full.size else np.zeros(nofder)
    result["P_DER_cur"] = p_der_full[:, 0] if p_der_full.size else np.zeros(nofder)
    result["R_DER_cur"] = r_der_full[:, 0] if r_der_full.size else np.zeros(nofder)

    result["Bid_R_rev"] = bid_r.copy()
    result["Bid_P_rev"] = bid_p.copy()
    result["P_DER_rev"] = p_der_full.copy()
    result["R_DER_rev"] = r_der_full.copy()
    result["E_rev"] = result["E_cur"].reshape(-1, 1)

    result["admm_bid_p_master"] = master_step.get("bid_p", bid_p).copy()
    result["admm_bid_r_master"] = master_step.get("bid_r", bid_r).copy()
    result["admm_deg_day"] = deg_day.copy()
    result["admm_user_solve_time_avg"] = float(np.mean(user_solve_time_history)) if user_solve_time_history else 0.0
    result["admm_user_solve_time_total"] = float(np.sum(user_solve_time_history))
    result["admm_user_solve_time_history"] = np.asarray(user_solve_time_history)
    result["admm_user_solve_time_avg_by_user"] = np.divide(
        user_solve_time_by_user,
        np.maximum(user_solve_counts_by_user, 1.0),
    )
    result["admm_system_time_avg"] = float(np.mean(system_solve_time_history)) if system_solve_time_history else 0.0
    result["admm_system_time_total"] = float(np.sum(system_solve_time_history))
    result["admm_system_time_history"] = np.asarray(system_solve_time_history)
    result["admm_ev_sum_p"] = master_step.get("z_p", np.zeros(nofslots)).copy()
    result["admm_ev_sum_r"] = master_step.get("z_r", np.zeros(nofslots)).copy()
    result["admm_ev_avg_p"] = result["admm_ev_sum_p"].copy()
    result["admm_ev_avg_r"] = result["admm_ev_sum_r"].copy()
    result["admm_primal_history"] = np.asarray(primal_history)
    result["admm_dual_history"] = np.asarray(dual_history)
    result["admm_eps_pri_history"] = np.asarray(eps_pri_history)
    result["admm_eps_dual_history"] = np.asarray(eps_dual_history)
    result["admm_iterations"] = len(primal_history)
    result["admm_converged"] = converged
    result["admm_ok"] = ok
    result["admm_central_indices"] = central_indices.copy()
    result["admm_ev_indices"] = ev_indices.copy()
    result["admm_ev_block_mode"] = "fleet"
