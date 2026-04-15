from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

try:
    from IPython import get_ipython
except ImportError:
    get_ipython = None

if get_ipython is None or get_ipython() is None:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from myexp.fast_control_implement import fast_control_implement
from myexp.main import _normalize_param, _normalize_param_std
from myexp.max_profit_1 import max_profit_1
from myexp.max_profit_t import max_profit_t
from myexp.data_process.config import ResourceConfig
from myexp.data_process.prepare_parameters import EV_TYPES, prepare_parameters
from myexp.data_process.prepare_price import prepare_price_data
from myexp.data_process.prepare_regd import prepare_regd_distribution
from myexp.data_process.prepare_std import prepare_std_parameters


matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["pdf.fonttype"] = 42


TYPE_LABELS = {
    "a": "A型(75kWh,11kW)",
    "b": "B型(60kWh,22kW)",
    "c": "C型(45kWh,7kW)",
}
TYPE_COLORS = {
    "a": "#1f77b4",
    "b": "#ff7f0e",
    "c": "#2ca02c",
}


def _dict_to_sns(d: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**d)


def _build_context(
    ev_count: int,
    aggregate_evs: bool,
    day_price: int = 21,
) -> tuple[dict[str, Any], Any]:
    config = ResourceConfig()
    config.ev.aggregate_evs = aggregate_evs
    config.ev.max_evs = ev_count

    nofslots = config.system.NOFSLOTS
    base_dir = project_root

    day_reg = day_price + 1

    hourly_distribution, hourly_mileage, d_s, signal_day = prepare_regd_distribution(
        day_reg=day_reg,
        NOFSLOTS=nofslots,
        granularity=0.1,
        nofHisDays=14,
        base_dir=base_dir,
    )
    price_reg, price_e, _ = prepare_price_data(
        day_price=day_price,
        hour_init=0,
        NOFSLOTS=nofslots,
        base_dir=base_dir,
    )

    param_resource, _ = prepare_parameters(
        NOFSLOTS=nofslots,
        base_dir=base_dir,
        config=config,
        target_ev_count=ev_count,
        ev_type_ratios=config.ev.type_ratios,
        ev_seed=config.ev.seed,
    )
    param_std = prepare_std_parameters(param_resource, NOFSLOTS=nofslots, config=config)

    param_market = _dict_to_sns(
        {
            "price_e": price_e,
            "price_reg": price_reg,
            "hourly_Mileage": hourly_mileage,
            "hourly_Distribution": hourly_distribution,
            "d_s": d_s,
            "s_perf": 0.984,
            "resource_names": ["pv", "es", "ev"],
        }
    )
    param_market = _normalize_param(param_market)
    param_std = _normalize_param_std(_dict_to_sns(param_std))

    nofder = 1 + 1 + param_resource.NOFEV
    nofscen = len(d_s)
    delta_t = 1.0
    delta_t_req = 0.5
    big_m = 1e6
    noftcap_bid = 1200
    noftcap_ctrl = 30

    result = {
        "Bid_P_day": np.zeros(nofslots),
        "Bid_R_day": np.zeros(nofslots),
        "P_DER_day": np.zeros((nofder, nofslots)),
        "R_DER_day": np.zeros((nofder, nofslots)),
        "E_day": np.zeros((nofder, nofslots + 1)),
        "Bid_P_rev": np.zeros(nofslots),
        "Bid_R_rev": np.zeros(nofslots),
        "P_DER_rev": np.zeros((nofder, nofslots)),
        "R_DER_rev": np.zeros((nofder, nofslots)),
        "E_rev": np.zeros((nofder, nofslots + 1)),
        "revision_times": np.zeros(nofslots),
        "P_alloc": np.zeros((nofder, 0)),
        "Signal_actual": np.zeros(nofslots * 1800 // noftcap_ctrl),
        "P_dis_actual": np.zeros((nofder, nofslots * 1800 // noftcap_ctrl)),
        "P_ch_actual": np.zeros((nofder, nofslots * 1800 // noftcap_ctrl)),
        "E_actual": np.zeros((nofder, nofslots * 1800 // noftcap_ctrl + 1)),
        "actualMil": np.zeros(nofslots),
        "actualEnergy": np.zeros(nofslots),
        "actualCost": np.zeros(nofslots),
        "EnegyFee_day": np.zeros(nofslots),
        "RegCapacity_day": np.zeros(nofslots),
        "RegMileage_day": np.zeros(nofslots),
        "Profit_day": np.zeros(nofslots),
        "Profit_rev": np.zeros(nofslots),
        "Profit_realtime": np.zeros(nofslots),
        "actualEnegyFee": np.zeros(nofslots),
        "actualRegCapacity": np.zeros(nofslots),
        "actualRegMileage": np.zeros(nofslots),
        "actualProfit": np.zeros(nofslots),
    }

    ctx = {
        "param": param_market,
        "param_std": param_std,
        "Signal_day": signal_day,
        "NOFSLOTS": nofslots,
        "NOFDER": nofder,
        "NOFSCEN": nofscen,
        "delta_t": delta_t,
        "delta_t_req": delta_t_req,
        "M": big_m,
        "NOFTCAP_ctrl": noftcap_ctrl,
        "result": result,
        "NOFTCAP_bid": noftcap_bid,
        "config": config,
        "day_price": day_price,
        "day_reg": day_reg,
    }
    return ctx, param_resource


def _run_experiment(
    ev_count: int,
    aggregate_evs: bool,
    day_price: int = 21,
) -> tuple[dict[str, Any], Any]:
    ctx, param_resource = _build_context(
        ev_count=ev_count,
        aggregate_evs=aggregate_evs,
        day_price=day_price,
    )
    param = ctx["param"]
    result = ctx["result"]
    nofslots = ctx["NOFSLOTS"]
    nofder = ctx["NOFDER"]
    delta_t = ctx["delta_t"]
    noftcap_bid = ctx["NOFTCAP_bid"]
    noftcap_ctrl = ctx["NOFTCAP_ctrl"]

    max_profit_1(ctx)

    result["Bid_P_day"] = result.get("Bid_P_init", np.zeros(nofslots)).copy()
    result["Bid_R_day"] = result.get("Bid_R_init", np.zeros(nofslots)).copy()
    result["P_DER_day"] = result.get("P_DER_rev", np.zeros((nofder, nofslots))).copy()
    result["R_DER_day"] = result.get("R_DER_rev", np.zeros((nofder, nofslots))).copy()
    result["E_day"] = result.get("E_init", np.zeros((nofder, nofslots + 1))).copy()

    result["EnegyFee_day"] = param.price_e * result["Bid_P_day"] * delta_t
    result["RegCapacity_day"] = param.price_reg[:, 0] * result["Bid_R_day"] * param.s_perf * delta_t
    result["RegMileage_day"] = (
        param.price_reg[:, 1] * param.hourly_Mileage * result["Bid_R_day"] * param.s_perf * delta_t
    )
    result["Profit_day"] = (
        result["EnegyFee_day"] + result["RegCapacity_day"] + result["RegMileage_day"]
    )

    for t_cap in range(1, (nofslots - 1) * 1800 + 1):
        if t_cap % noftcap_bid == 1:
            delta_t_rest = delta_t - ((t_cap - 1) % 1800) / 1800.0
            ctx.update({"t_cap": t_cap, "delta_t_rest": delta_t_rest})
            max_profit_t(ctx)
        if t_cap % noftcap_ctrl == 1:
            ctx.update({"t_cap": t_cap})
            fast_control_implement(ctx)

    for t_cap in range((nofslots - 1) * 1800 + 1, nofslots * 1800):
        if t_cap % noftcap_ctrl == 1:
            ctx.update({"t_cap": t_cap})
            fast_control_implement(ctx)

    result["actualEnegyFee"] = param.price_e * result["actualEnergy"]
    result["actualRegCapacity"] = param.price_reg[:, 0] * result["Bid_R_rev"] * param.s_perf * delta_t
    result["actualRegMileage"] = (param.price_reg[:, 1] * result["actualMil"]) * param.s_perf * delta_t
    result["actualProfit"] = (
        result["actualEnegyFee"] + result["actualRegCapacity"] + result["actualRegMileage"]
    )
    return ctx, param_resource


def _schedule_dataframe(param_resource: Any) -> pd.DataFrame:
    u = np.asarray(param_resource.u)
    nofslots = u.shape[1]
    arrivals = np.argmax(u > 0, axis=1) + 1
    departures = nofslots - np.argmax(np.flip(u, axis=1) > 0, axis=1)
    connected_slots = u.sum(axis=1).astype(int)

    df = pd.DataFrame(
        {
            "ev_id": [f"EV_{idx + 1:03d}" for idx in range(u.shape[0])],
            "type": param_resource.ev_types,
            "type_label": [TYPE_LABELS.get(t, str(t)) for t in param_resource.ev_types],
            "arrival_slot": arrivals.astype(int),
            "departure_slot": departures.astype(int),
            "connected_slots": connected_slots,
            "battery_kwh": [EV_TYPES[t].battery_capacity for t in param_resource.ev_types],
            "charge_kw": [EV_TYPES[t].power_ch_limit for t in param_resource.ev_types],
            "target_soc_pct": [EV_TYPES[t].energy_end_ratio * 100.0 for t in param_resource.ev_types],
        }
    )
    return df


def _slot_summary_dataframe(result: dict[str, Any], schedule_df: pd.DataFrame, nofslots: int) -> pd.DataFrame:
    slot_index = np.arange(1, nofslots + 1)
    arrivals = (
        schedule_df.groupby("arrival_slot").size().reindex(slot_index, fill_value=0).astype(int).to_numpy()
    )
    departures = (
        schedule_df.groupby("departure_slot").size().reindex(slot_index, fill_value=0).astype(int).to_numpy()
    )
    online = np.zeros(nofslots, dtype=int)
    for _, row in schedule_df.iterrows():
        online[row["arrival_slot"] - 1 : row["departure_slot"]] += 1

    bid_p_day = np.asarray(result["Bid_P_day"], dtype=float)
    bid_p_rev = np.asarray(result["Bid_P_rev"], dtype=float)
    bid_r_day = np.asarray(result["Bid_R_day"], dtype=float)
    bid_r_rev = np.asarray(result["Bid_R_rev"], dtype=float)

    df = pd.DataFrame(
        {
            "slot": slot_index,
            "arrivals": arrivals,
            "departures": departures,
            "online_ev_count": online,
            "bid_p_day_mw": bid_p_day,
            "bid_p_corrected_mw": bid_p_rev,
            "bid_p_delta_mw": bid_p_rev - bid_p_day,
            "bid_r_day_mw": bid_r_day,
            "bid_r_corrected_mw": bid_r_rev,
            "bid_r_delta_mw": bid_r_rev - bid_r_day,
            "day_profit_usd": np.asarray(result["Profit_day"], dtype=float),
            "actual_profit_usd": np.asarray(result["actualProfit"], dtype=float),
        }
    )
    return df


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _write_schedule_plot(schedule_df: pd.DataFrame, slot_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

    rng = np.random.default_rng(42)
    ax = axes[0]
    for ev_type in ["a", "b", "c"]:
        subset = schedule_df[schedule_df["type"] == ev_type]
        if subset.empty:
            continue
        jitter_x = rng.uniform(-0.08, 0.08, len(subset))
        jitter_y = rng.uniform(-0.08, 0.08, len(subset))
        ax.scatter(
            subset["arrival_slot"] + jitter_x,
            subset["departure_slot"] + jitter_y,
            s=32,
            alpha=0.7,
            color=TYPE_COLORS[ev_type],
            label=TYPE_LABELS.get(ev_type, ev_type),
        )
    ax.plot([1, slot_df["slot"].max()], [1, slot_df["slot"].max()], "--", color="#999999", linewidth=1.0)
    ax.set_title("EV异质到达/离开时间散点")
    ax.set_xlabel("到达时段")
    ax.set_ylabel("离开时段")
    ax.set_xlim(0.5, slot_df["slot"].max() + 0.5)
    ax.set_ylim(0.5, slot_df["slot"].max() + 0.5)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.bar(
        slot_df["slot"],
        slot_df["arrivals"],
        color="#7db7e8",
        width=0.75,
        label="到达数量",
    )
    ax.bar(
        slot_df["slot"],
        -slot_df["departures"],
        color="#f3a6a0",
        width=0.75,
        label="离开数量",
    )
    ax.axhline(0.0, color="#666666", linewidth=1.0)
    ax.set_title("每时段到达/离开数量")
    ax.set_xlabel("时段")
    ax.set_ylabel("车辆数")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax2 = ax.twinx()
    ax2.plot(
        slot_df["slot"],
        slot_df["online_ev_count"],
        color="#2ca25f",
        marker="o",
        linewidth=2.0,
        label="在线EV数量",
    )
    ax2.set_ylabel("在线数量")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_bid_plot(slot_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8.2), sharex=True)

    ax = axes[0]
    ax.plot(
        slot_df["slot"],
        slot_df["bid_p_day_mw"],
        color="#1f4e79",
        linewidth=2.2,
        marker="o",
        label="修正前 bid_p",
    )
    ax.plot(
        slot_df["slot"],
        slot_df["bid_p_corrected_mw"],
        color="#d95f02",
        linewidth=2.2,
        marker="s",
        label="修正后 bid_p",
    )
    ax.set_ylabel("bid_p (MW)")
    ax.set_title("bid_p 修正前后对比")
    ax.grid(alpha=0.25, linestyle=":")
    ax2 = ax.twinx()
    ax2.fill_between(
        slot_df["slot"],
        0,
        slot_df["online_ev_count"],
        color="#74c476",
        alpha=0.18,
        label="在线EV数量",
    )
    ax2.set_ylabel("在线EV数量")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper right")

    ax = axes[1]
    delta = slot_df["bid_p_delta_mw"]
    colors = np.where(delta >= 0, "#ef8a62", "#67a9cf")
    ax.bar(slot_df["slot"], delta, color=colors, width=0.72, label="修正幅度")
    ax.plot(
        slot_df["slot"],
        slot_df["departures"],
        color="#7f0000",
        linewidth=1.8,
        marker="^",
        label="离开数量",
    )
    ax.plot(
        slot_df["slot"],
        slot_df["arrivals"],
        color="#2166ac",
        linewidth=1.8,
        marker="v",
        label="到达数量",
    )
    ax.axhline(0.0, color="#666666", linewidth=1.0)
    ax.set_xlabel("时段")
    ax.set_ylabel("修正后 - 修正前 (MW)")
    ax.set_title("bid_p 修正幅度与到离站变化")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_summary_text(
    summary_path: Path,
    schedule_df: pd.DataFrame,
    slot_df: pd.DataFrame,
    result: dict[str, Any],
    aggregate_evs: bool,
) -> str:
    type_counts = Counter(schedule_df["type"])
    peak_arrival_slot = int(slot_df.loc[slot_df["arrivals"].idxmax(), "slot"])
    peak_departure_slot = int(slot_df.loc[slot_df["departures"].idxmax(), "slot"])
    peak_online_slot = int(slot_df.loc[slot_df["online_ev_count"].idxmax(), "slot"])

    mean_abs_bid_delta = float(np.abs(slot_df["bid_p_delta_mw"]).mean())
    max_abs_idx = int(np.abs(slot_df["bid_p_delta_mw"]).idxmax())
    max_abs_slot = int(slot_df.loc[max_abs_idx, "slot"])
    corr_abs_delta_depart = _safe_corr(
        np.abs(slot_df["bid_p_delta_mw"]).to_numpy(),
        slot_df["departures"].to_numpy(),
    )
    corr_abs_delta_online = _safe_corr(
        np.abs(slot_df["bid_p_delta_mw"]).to_numpy(),
        slot_df["online_ev_count"].to_numpy(),
    )
    top_slots = slot_df.reindex(
        np.argsort(np.abs(slot_df["bid_p_delta_mw"]).to_numpy())[::-1][:5]
    )[["slot", "bid_p_delta_mw", "online_ev_count", "arrivals", "departures"]]

    lines = [
        "异质EV到离站与 bid_p 修正结果分析",
        "=" * 72,
        "",
        "[实验设置]",
        f"- EV数量: {len(schedule_df)} 辆",
        f"- 是否聚合求解: {'是' if aggregate_evs else '否'}",
        f"- 时段数: {len(slot_df)}",
        f"- 类型分布: A型 {type_counts.get('a', 0)} 辆, B型 {type_counts.get('b', 0)} 辆, C型 {type_counts.get('c', 0)} 辆",
        "",
        "[到离站异质性]",
        f"- 最早到达时段: {int(schedule_df['arrival_slot'].min())}",
        f"- 最晚离开时段: {int(schedule_df['departure_slot'].max())}",
        f"- 到达高峰时段: {peak_arrival_slot} (到达 {int(slot_df['arrivals'].max())} 辆)",
        f"- 离开高峰时段: {peak_departure_slot} (离开 {int(slot_df['departures'].max())} 辆)",
        f"- 在线数量峰值时段: {peak_online_slot} (在线 {int(slot_df['online_ev_count'].max())} 辆)",
        "",
        "[bid_p 修正效果]",
        f"- 日前 bid_p 总量: {float(slot_df['bid_p_day_mw'].sum()):.4f} MW",
        f"- 修正后 bid_p 总量: {float(slot_df['bid_p_corrected_mw'].sum()):.4f} MW",
        f"- 平均绝对修正幅度: {mean_abs_bid_delta:.4f} MW",
        f"- 最大绝对修正时段: {max_abs_slot} (Δbid_p = {float(slot_df.loc[max_abs_idx, 'bid_p_delta_mw']):+.4f} MW)",
        f"- |Δbid_p| 与离开数量相关系数: {corr_abs_delta_depart:.4f}",
        f"- |Δbid_p| 与在线EV数量相关系数: {corr_abs_delta_online:.4f}",
        "",
        "[利润对比]",
        f"- 日前总利润: {float(np.sum(result['Profit_day'])):.4f} USD",
        f"- 实际总利润: {float(np.sum(result['actualProfit'])):.4f} USD",
        f"- 利润变化: {float(np.sum(result['actualProfit']) - np.sum(result['Profit_day'])):+.4f} USD",
        "",
        "[修正最明显的5个时段]",
    ]
    for _, row in top_slots.iterrows():
        lines.append(
            "- 时段 {slot}: Δbid_p={delta:+.4f} MW, 在线EV={online}, 到达={arr}, 离开={dep}".format(
                slot=int(row["slot"]),
                delta=float(row["bid_p_delta_mw"]),
                online=int(row["online_ev_count"]),
                arr=int(row["arrivals"]),
                dep=int(row["departures"]),
            )
        )
    lines += [
        "",
        "[结论解释]",
        "1. 车辆到达/离开时间并不一致，导致 EV 在线数量在全天内明显波动，可用于投标的可调容量不是常数。",
        "2. bid_p 的修正主要集中在到离站变化更剧烈的时段，尤其是离站高峰前后，因为此时 EV 需要优先满足离站能量约束。",
        "3. 当在线车辆减少时，修正后的 bid_p 往往更保守；当大量车辆新接入时，聚合器才有空间重新抬高可调功率。",
        "4. 因此，用异质到离站建模后，修正机制的作用不只是“数值微调”，而是在真实可用资源变化下把投标重新拉回可执行范围。",
        "",
    ]

    text = "\n".join(lines)
    summary_path.write_text(text, encoding="utf-8")
    return text


def main(ev_count: int = 360, aggregate_evs: bool = False, day_price: int = 21) -> None:
    ctx, param_resource = _run_experiment(
        ev_count=ev_count,
        aggregate_evs=aggregate_evs,
        day_price=day_price,
    )
    result = ctx["result"]
    nofslots = ctx["NOFSLOTS"]

    schedule_df = _schedule_dataframe(param_resource)
    slot_df = _slot_summary_dataframe(result, schedule_df, nofslots)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(project_root) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    schedule_csv = results_dir / f"heterogeneous_ev_schedule_{ev_count}_{timestamp}.csv"
    slot_csv = results_dir / f"heterogeneous_bid_slot_summary_{ev_count}_{timestamp}.csv"
    schedule_plot = results_dir / f"heterogeneous_ev_schedule_{ev_count}_{timestamp}.png"
    bid_plot = results_dir / f"heterogeneous_bid_comparison_{ev_count}_{timestamp}.png"
    summary_txt = results_dir / f"heterogeneous_bid_analysis_{ev_count}_{timestamp}.txt"

    schedule_df.to_csv(schedule_csv, index=False, encoding="utf-8-sig")
    slot_df.to_csv(slot_csv, index=False, encoding="utf-8-sig")
    _write_schedule_plot(schedule_df, slot_df, schedule_plot)
    _write_bid_plot(slot_df, bid_plot)
    summary_text = _write_summary_text(summary_txt, schedule_df, slot_df, result, aggregate_evs)

    print("\n实验完成")
    print(f"schedule_csv={schedule_csv}")
    print(f"slot_csv={slot_csv}")
    print(f"schedule_plot={schedule_plot}")
    print(f"bid_plot={bid_plot}")
    print(f"summary_txt={summary_txt}")
    print("\n关键摘要:")
    print(summary_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="heterogeneous EV bid correction analysis")
    parser.add_argument("--ev_count", type=int, default=360, help="EV count")
    parser.add_argument(
        "--aggregate_evs",
        action="store_true",
        help="Aggregate EVs by (arrival, departure, type) before solving",
    )
    parser.add_argument(
        "--day_price",
        type=int,
        default=21,
        help="Price day index; regulation signal uses the next day by default",
    )
    args = parser.parse_args()
    main(ev_count=args.ev_count, aggregate_evs=args.aggregate_evs, day_price=args.day_price)