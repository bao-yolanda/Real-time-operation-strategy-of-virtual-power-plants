from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from myexp.heterogeneous_ev_bid_analysis import _build_context, _run_experiment


def _choose_solver(preferred: str) -> str:
    installed = cp.installed_solvers()
    if preferred in installed:
        return preferred
    for fallback in ["ECOS", "OSQP", "SCS"]:
        if fallback in installed:
            return fallback
    return "SCS"


def _solve_energy_only(ctx: dict[str, Any]) -> dict[str, Any]:
    """只参与电量市场的确定性基线模型。"""
    param = ctx["param"]
    param_std = ctx["param_std"]
    nofslots = int(ctx["NOFSLOTS"])
    nofder = int(ctx["NOFDER"])
    delta_t = float(ctx["delta_t"])

    price_e = np.asarray(param.price_e, dtype=float)
    energy_init = np.asarray(param_std.energy_init, dtype=float)
    energy_upper = np.asarray(param_std.energy_upper_limit, dtype=float)
    energy_lower = np.asarray(param_std.energy_lower_limit, dtype=float)
    power_dis_upper = np.asarray(param_std.power_dis_upper_limit, dtype=float)
    power_dis_lower = np.asarray(param_std.power_dis_lower_limit, dtype=float)
    power_ch_upper = np.asarray(param_std.power_ch_upper_limit, dtype=float)
    power_ch_lower = np.asarray(param_std.power_ch_lower_limit, dtype=float)
    theta = np.asarray(param_std.theta, dtype=float)
    eta_ch = np.asarray(param_std.eta_ch, dtype=float)
    eta_dis = np.asarray(param_std.eta_dis, dtype=float)
    pr_dis = np.asarray(param_std.pr_dis, dtype=float)
    pr_ch = np.asarray(param_std.pr_ch, dtype=float)
    w_omega = np.asarray(param_std.wOmiga, dtype=float)

    bid_p = cp.Variable(nofslots)
    p_der = cp.Variable((nofder, nofslots))
    p_dis = cp.Variable((nofder, nofslots))
    p_ch = cp.Variable((nofder, nofslots))
    energy = cp.Variable((nofder, nofslots + 1))

    energy_fee = cp.multiply(price_e, bid_p) * delta_t
    battery_deg = (
        cp.sum(
            cp.multiply(pr_dis[:, None], p_dis) + cp.multiply(pr_ch[:, None], p_ch),
            axis=0,
        )
        * delta_t
    )
    profit = cp.sum(energy_fee - battery_deg)

    constraints: list[cp.Constraint] = [
        energy[:, 0] == energy_init,
        p_der == p_dis - p_ch,
        bid_p == cp.sum(p_der, axis=0),
        p_dis >= power_dis_lower,
        p_dis <= power_dis_upper,
        p_ch >= power_ch_lower,
        p_ch <= power_ch_upper,
        energy[:, 1:] >= energy_lower,
        energy[:, 1:] <= energy_upper,
        energy[:, 1:]
        == cp.multiply(theta[:, None], energy[:, :-1])
        + (eta_ch @ p_ch) * delta_t
        - (eta_dis @ p_dis) * delta_t
        + w_omega * delta_t,
    ]

    problem = cp.Problem(cp.Maximize(profit), constraints)
    solver_name = _choose_solver("GUROBI")
    try:
        problem.solve(solver=solver_name, verbose=False)
    except Exception:
        solver_name = _choose_solver("ECOS")
        problem.solve(solver=solver_name, verbose=False)

    ok = problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    if not ok or bid_p.value is None:
        raise RuntimeError(f"电量市场基线求解失败: status={problem.status}")

    bid_p_val = np.asarray(bid_p.value, dtype=float)
    p_der_val = np.asarray(p_der.value, dtype=float)
    p_dis_val = np.asarray(p_dis.value, dtype=float)
    p_ch_val = np.asarray(p_ch.value, dtype=float)
    energy_val = np.asarray(energy.value, dtype=float)
    energy_fee_val = price_e * bid_p_val * delta_t
    battery_deg_val = (
        np.sum(pr_dis[:, None] * p_dis_val + pr_ch[:, None] * p_ch_val, axis=0) * delta_t
    )
    profit_val = energy_fee_val - battery_deg_val

    return {
        "bid_p_mw": bid_p_val,
        "p_der_mw": p_der_val,
        "p_dis_mw": p_dis_val,
        "p_ch_mw": p_ch_val,
        "energy_state_mwh": energy_val,
        "energy_fee_usd": energy_fee_val,
        "battery_deg_usd": battery_deg_val,
        "profit_usd": profit_val,
        "total_profit_usd": float(np.sum(profit_val)),
        "total_energy_fee_usd": float(np.sum(energy_fee_val)),
        "total_battery_deg_usd": float(np.sum(battery_deg_val)),
        "solver_status": problem.status,
    }


def _recompute_full_market_metrics(ctx: dict[str, Any]) -> dict[str, np.ndarray | float]:
    """基于现有复杂模型的原始输出，统一按净收益口径重算一次。"""
    param = ctx["param"]
    result = ctx["result"]
    delta_t = float(ctx["delta_t"])

    actual_energy_fee = np.asarray(param.price_e, dtype=float) * np.asarray(
        result["actualEnergy"], dtype=float
    )
    actual_reg_capacity = (
        np.asarray(param.price_reg[:, 0], dtype=float)
        * np.asarray(result["Bid_R_rev"], dtype=float)
        * float(param.s_perf)
        * delta_t
    )
    actual_reg_mileage = (
        np.asarray(param.price_reg[:, 1], dtype=float)
        * np.asarray(result["actualMil"], dtype=float)
        * float(param.s_perf)
        * delta_t
    )
    actual_battery_deg = np.asarray(result["actualCost"], dtype=float)
    actual_profit = actual_energy_fee + actual_reg_capacity + actual_reg_mileage - actual_battery_deg

    return {
        "bid_p_mw": np.asarray(result["Bid_P_rev"], dtype=float),
        "bid_r_mw": np.asarray(result["Bid_R_rev"], dtype=float),
        "energy_fee_usd": actual_energy_fee,
        "reg_capacity_usd": actual_reg_capacity,
        "reg_mileage_usd": actual_reg_mileage,
        "battery_deg_usd": actual_battery_deg,
        "profit_usd": actual_profit,
        "total_profit_usd": float(np.sum(actual_profit)),
        "total_energy_fee_usd": float(np.sum(actual_energy_fee)),
        "total_reg_capacity_usd": float(np.sum(actual_reg_capacity)),
        "total_reg_mileage_usd": float(np.sum(actual_reg_mileage)),
        "total_battery_deg_usd": float(np.sum(actual_battery_deg)),
    }


def _build_slot_dataframe(
    energy_only: dict[str, Any],
    full_market: dict[str, Any] | None,
    nofslots: int,
) -> pd.DataFrame:
    slot = np.arange(1, nofslots + 1, dtype=int)
    df = pd.DataFrame(
        {
            "slot": slot,
            "energy_only_bid_p_mw": np.asarray(energy_only["bid_p_mw"], dtype=float),
            "energy_only_energy_fee_usd": np.asarray(energy_only["energy_fee_usd"], dtype=float),
            "energy_only_battery_deg_usd": np.asarray(energy_only["battery_deg_usd"], dtype=float),
            "energy_only_profit_usd": np.asarray(energy_only["profit_usd"], dtype=float),
        }
    )
    if full_market is not None:
        df["full_market_bid_p_mw"] = np.asarray(full_market["bid_p_mw"], dtype=float)
        df["full_market_bid_r_mw"] = np.asarray(full_market["bid_r_mw"], dtype=float)
        df["full_market_energy_fee_usd"] = np.asarray(full_market["energy_fee_usd"], dtype=float)
        df["full_market_reg_capacity_usd"] = np.asarray(full_market["reg_capacity_usd"], dtype=float)
        df["full_market_reg_mileage_usd"] = np.asarray(full_market["reg_mileage_usd"], dtype=float)
        df["full_market_battery_deg_usd"] = np.asarray(full_market["battery_deg_usd"], dtype=float)
        df["full_market_profit_usd"] = np.asarray(full_market["profit_usd"], dtype=float)
        df["profit_gap_usd"] = df["full_market_profit_usd"] - df["energy_only_profit_usd"]
    return df


def _build_summary_text(
    ev_count: int,
    aggregate_evs: bool,
    day_price: int,
    energy_only: dict[str, Any],
    full_market: dict[str, Any] | None,
) -> str:
    lines = [
        "电量市场单独参与 vs 能量+调频联合参与 对比",
        "=" * 72,
        "",
        "[实验设置]",
        f"- EV数量: {ev_count}",
        f"- 是否聚合EV求解: {'是' if aggregate_evs else '否'}",
        f"- 电价日: day {day_price}",
        f"- 调频信号日: day {day_price + 1}",
        "- 基线模型: 仅参与电量市场，不参与调频，不做日前/日内滚动修正",
        "- 对比模型: 复用现有完整模型（能量市场 + 调频市场 + 修正 + 实时执行）",
        "",
        "[仅电量市场基线]",
        f"- 总净收益: {energy_only['total_profit_usd']:.4f} USD",
        f"- 能量收益: {energy_only['total_energy_fee_usd']:.4f} USD",
        f"- 电池退化费用: {energy_only['total_battery_deg_usd']:.4f} USD",
    ]
    if full_market is not None:
        extra_profit = full_market["total_profit_usd"] - energy_only["total_profit_usd"]
        lines += [
            "",
            "[完整市场模型]",
            f"- 总净收益: {full_market['total_profit_usd']:.4f} USD",
            f"- 能量收益: {full_market['total_energy_fee_usd']:.4f} USD",
            f"- 调频容量收益: {full_market['total_reg_capacity_usd']:.4f} USD",
            f"- 调频里程收益: {full_market['total_reg_mileage_usd']:.4f} USD",
            f"- 电池退化费用: {full_market['total_battery_deg_usd']:.4f} USD",
            "",
            "[对比结论]",
            f"- 调频带来的净增收益: {extra_profit:+.4f} USD",
            "- 若该值为正，说明调频市场在扣除额外退化成本后仍提高了总收益。",
            "- 若该值接近0或为负，说明在当前价格和资源条件下，只做电量市场可能更划算。",
            "",
        ]
    return "\n".join(lines)


def main(
    ev_count: int = 120,
    aggregate_evs: bool = False,
    skip_full_market: bool = False,
    day_price: int = 15,
) -> None:
    energy_ctx, _ = _build_context(
        ev_count=ev_count,
        aggregate_evs=aggregate_evs,
        day_price=day_price,
    )
    energy_only = _solve_energy_only(energy_ctx)

    full_market = None
    if not skip_full_market:
        full_ctx, _ = _run_experiment(
            ev_count=ev_count,
            aggregate_evs=aggregate_evs,
            day_price=day_price,
        )
        full_market = _recompute_full_market_metrics(full_ctx)

    nofslots = int(energy_ctx["NOFSLOTS"])
    slot_df = _build_slot_dataframe(energy_only, full_market, nofslots)
    summary_text = _build_summary_text(
        ev_count=ev_count,
        aggregate_evs=aggregate_evs,
        day_price=day_price,
        energy_only=energy_only,
        full_market=full_market,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(project_root) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    slot_csv = results_dir / f"energy_only_vs_full_market_day{day_price}_{ev_count}_{timestamp}.csv"
    summary_txt = results_dir / f"energy_only_vs_full_market_day{day_price}_{ev_count}_{timestamp}.txt"
    slot_df.to_csv(slot_csv, index=False, encoding="utf-8-sig")
    summary_txt.write_text(summary_text, encoding="utf-8")

    print(summary_text)
    print(f"时段级结果已保存到: {slot_csv}")
    print(f"汇总说明已保存到: {summary_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="仅电量市场基线与完整市场模型收益对比")
    parser.add_argument("--ev_count", type=int, default=120, help="EV数量，默认120")
    parser.add_argument(
        "--day_price",
        type=int,
        default=15,
        help="电价日编号；调频信号默认使用下一天",
    )
    parser.add_argument(
        "--aggregate_evs",
        action="store_true",
        help="是否对相同连接模式的EV做聚合求解",
    )
    parser.add_argument(
        "--skip_full_market",
        action="store_true",
        help="只运行电量市场基线，不运行完整市场模型",
    )
    args = parser.parse_args()
    main(
        ev_count=args.ev_count,
        aggregate_evs=args.aggregate_evs,
        skip_full_market=args.skip_full_market,
        day_price=args.day_price,
    )
