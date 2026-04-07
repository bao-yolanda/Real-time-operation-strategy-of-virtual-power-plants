from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def resolve_project_root(start: Path | None = None) -> Path:
    """向上查找包含 myexp 目录的项目根目录。"""
    if start is None:
        start = Path.cwd()

    for candidate in [start, *start.parents]:
        if (candidate / "myexp").exists():
            return candidate

    raise RuntimeError("未找到包含 'myexp' 目录的项目根目录。")


def load_or_run_simulation(
    project_root: Path,
    cache_path: Path,
    force_rerun: bool = False,
    data_source: str = "data_process",
    mat_file: str | None = None,
    save_results: bool = False,
    day_price: int = 25,
) -> dict[str, Any]:
    """优先从缓存读取；没有缓存时调用 myexp.main.main。"""
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force_rerun:
        with cache_path.open("rb") as fh:
            return pickle.load(fh)

    from myexp.main import main

    sim = main(data_source=data_source, mat_file=mat_file, save_results=save_results, day_price=day_price)
    with cache_path.open("wb") as fh:
        pickle.dump(sim, fh)
    return sim


def _slot_hours(nofslots: int) -> np.ndarray:
    return np.arange(nofslots, dtype=float)


def _slot_numbers(nofslots: int) -> np.ndarray:
    return np.arange(1, nofslots + 1, dtype=int)


def _ev_capacity(meta: dict[str, Any], ev_pos: int) -> float:
    upper = np.asarray(meta["ev_energy_upper"][ev_pos], dtype=float)
    positive = upper[upper > 0]
    if positive.size == 0:
        return 1.0
    return float(positive.max())


def build_summary_df(sim: dict[str, Any]) -> pd.DataFrame:
    result = sim["result"]
    meta = sim["resource_meta"]

    summary_items = [
        ("时段数", sim["NOFSLOTS"]),
        ("资源总数", sim["NOFDER"]),
        ("EV数量", len(meta["ev_labels"])),
        ("日前利润(USD)", float(np.sum(result["Profit_day"]))),
        ("实际利润(USD)", float(np.sum(result["actualProfit"]))),
        ("日前电池退化费用(USD)", float(np.sum(result.get("BatteryDeg_day", np.zeros(sim["NOFSLOTS"]))))),
        ("实际电池退化费用(USD)", float(np.sum(result.get("actualBatteryDeg", result["actualCost"])))),
        ("利润增量(USD)", float(np.sum(result["actualProfit"]) - np.sum(result["Profit_day"]))),
        ("利润增量(%)", float((np.sum(result["actualProfit"]) - np.sum(result["Profit_day"])) / np.sum(result["Profit_day"]) * 100.0)),
        ("bid_p平均修正(MW)", float(np.mean(result["Bid_P_rev"] - result["Bid_P_day"]))),
        ("bid_r平均修正(MW)", float(np.mean(result["Bid_R_rev"] - result["Bid_R_day"]))),
        ("滚动优化次数", float(np.sum(result["revision_times"]))),
    ]
    return pd.DataFrame(summary_items, columns=["指标", "数值"])


def build_slot_df(sim: dict[str, Any]) -> pd.DataFrame:
    param = sim["param"]
    result = sim["result"]
    nofslots = sim["NOFSLOTS"]

    slot_df = pd.DataFrame(
        {
            "slot": _slot_numbers(nofslots),
            "hour": _slot_hours(nofslots),
            "bid_p_day": np.asarray(result["Bid_P_day"], dtype=float),
            "bid_p_rev": np.asarray(result["Bid_P_rev"], dtype=float),
            "bid_r_day": np.asarray(result["Bid_R_day"], dtype=float),
            "bid_r_rev": np.asarray(result["Bid_R_rev"], dtype=float),
            "profit_day": np.asarray(result["Profit_day"], dtype=float),
            "profit_actual": np.asarray(result["actualProfit"], dtype=float),
            "battery_deg_day": np.asarray(result.get("BatteryDeg_day", np.zeros(nofslots)), dtype=float),
            "battery_deg_actual": np.asarray(result.get("actualBatteryDeg", result["actualCost"]), dtype=float),
            "actual_mileage": np.asarray(result["actualMil"], dtype=float),
            "actual_energy": np.asarray(result["actualEnergy"], dtype=float),
            "actual_cost": np.asarray(result["actualCost"], dtype=float),
            "revision_count": np.asarray(result["revision_times"], dtype=float),
            "price_e": np.asarray(param.price_e, dtype=float),
            "price_reg_cap": np.asarray(param.price_reg[:, 0], dtype=float),
            "price_reg_mileage": np.asarray(param.price_reg[:, 1], dtype=float),
            "hourly_mileage": np.asarray(param.hourly_Mileage, dtype=float),
        }
    )
    slot_df["bid_p_delta"] = slot_df["bid_p_rev"] - slot_df["bid_p_day"]
    slot_df["bid_r_delta"] = slot_df["bid_r_rev"] - slot_df["bid_r_day"]
    slot_df["profit_delta"] = slot_df["profit_actual"] - slot_df["profit_day"]
    return slot_df


def build_resource_df(sim: dict[str, Any]) -> pd.DataFrame:
    result = sim["result"]
    nofslots = sim["NOFSLOTS"]

    resource_rows: list[dict[str, Any]] = []
    ev_slice = slice(2, None)

    for slot_idx in range(nofslots):
        resource_rows.extend(
            [
                {
                    "slot": slot_idx + 1,
                    "resource_type": "PV",
                    "bid_p_day": float(result["P_DER_day"][0, slot_idx]),
                    "bid_p_rev": float(result["P_DER_rev"][0, slot_idx]),
                    "bid_r_day": float(result["R_DER_day"][0, slot_idx]),
                    "bid_r_rev": float(result["R_DER_rev"][0, slot_idx]),
                },
                {
                    "slot": slot_idx + 1,
                    "resource_type": "ES",
                    "bid_p_day": float(result["P_DER_day"][1, slot_idx]),
                    "bid_p_rev": float(result["P_DER_rev"][1, slot_idx]),
                    "bid_r_day": float(result["R_DER_day"][1, slot_idx]),
                    "bid_r_rev": float(result["R_DER_rev"][1, slot_idx]),
                },
                {
                    "slot": slot_idx + 1,
                    "resource_type": "EV",
                    "bid_p_day": float(np.sum(result["P_DER_day"][ev_slice, slot_idx])),
                    "bid_p_rev": float(np.sum(result["P_DER_rev"][ev_slice, slot_idx])),
                    "bid_r_day": float(np.sum(result["R_DER_day"][ev_slice, slot_idx])),
                    "bid_r_rev": float(np.sum(result["R_DER_rev"][ev_slice, slot_idx])),
                },
            ]
        )

    resource_df = pd.DataFrame(resource_rows)
    resource_df["bid_p_delta"] = resource_df["bid_p_rev"] - resource_df["bid_p_day"]
    resource_df["bid_r_delta"] = resource_df["bid_r_rev"] - resource_df["bid_r_day"]
    return resource_df


def build_ev_heatmap_data(sim: dict[str, Any]) -> dict[str, Any]:
    result = sim["result"]
    meta = sim["resource_meta"]
    nofslots = sim["NOFSLOTS"]
    ev_count = len(meta["ev_labels"])

    if ev_count == 0:
        empty = np.zeros((0, nofslots))
        return {
            "steps_per_slot": 0,
            "bid_p_day": empty,
            "bid_p_rev": empty,
            "bid_p_delta": empty,
            "bid_r_day": empty,
            "bid_r_rev": empty,
            "bid_r_delta": empty,
            "actual_net_hourly": empty,
            "soc_hourly": empty,
            "availability": np.zeros((0, nofslots), dtype=bool),
        }

    ev_indices = meta["ev_indices"]
    steps_per_slot = result["P_dis_actual"].shape[1] // nofslots
    actual_net = np.asarray(result["P_dis_actual"][ev_indices] - result["P_ch_actual"][ev_indices], dtype=float)
    actual_net_hourly = actual_net.reshape(ev_count, nofslots, steps_per_slot).mean(axis=2)

    soc_hourly = np.zeros((ev_count, nofslots), dtype=float)
    energy_actual = np.asarray(result["E_actual"][ev_indices], dtype=float)
    for ev_pos in range(ev_count):
        capacity = _ev_capacity(meta, ev_pos)
        hour_end_energy = energy_actual[ev_pos, steps_per_slot::steps_per_slot]
        soc_hourly[ev_pos] = hour_end_energy[:nofslots] / capacity * 100.0

    return {
        "steps_per_slot": steps_per_slot,
        "bid_p_day": np.asarray(result["P_DER_day"][ev_indices], dtype=float),
        "bid_p_rev": np.asarray(result["P_DER_rev"][ev_indices], dtype=float),
        "bid_p_delta": np.asarray(result["P_DER_rev"][ev_indices] - result["P_DER_day"][ev_indices], dtype=float),
        "bid_r_day": np.asarray(result["R_DER_day"][ev_indices], dtype=float),
        "bid_r_rev": np.asarray(result["R_DER_rev"][ev_indices], dtype=float),
        "bid_r_delta": np.asarray(result["R_DER_rev"][ev_indices] - result["R_DER_day"][ev_indices], dtype=float),
        "actual_net_hourly": actual_net_hourly,
        "soc_hourly": soc_hourly,
        "availability": np.asarray(meta["ev_available"], dtype=bool),
    }


def build_selected_ev_slot_df(sim: dict[str, Any], ev_number: int) -> pd.DataFrame:
    result = sim["result"]
    meta = sim["resource_meta"]
    nofslots = sim["NOFSLOTS"]

    ev_pos = ev_number - 1
    if ev_pos < 0 or ev_pos >= len(meta["ev_labels"]):
        raise IndexError(f"ev_number={ev_number} 超出范围。")

    ev_idx = int(meta["ev_indices"][ev_pos])
    capacity = _ev_capacity(meta, ev_pos)

    slot_df = pd.DataFrame(
        {
            "slot": _slot_numbers(nofslots),
            "hour": _slot_hours(nofslots),
            "available": np.asarray(meta["ev_available"][ev_pos], dtype=bool),
            "bid_p_day": np.asarray(result["P_DER_day"][ev_idx], dtype=float),
            "bid_p_rev": np.asarray(result["P_DER_rev"][ev_idx], dtype=float),
            "bid_r_day": np.asarray(result["R_DER_day"][ev_idx], dtype=float),
            "bid_r_rev": np.asarray(result["R_DER_rev"][ev_idx], dtype=float),
            "energy_day_start": np.asarray(result["E_day"][ev_idx, :-1], dtype=float),
            "energy_day_end": np.asarray(result["E_day"][ev_idx, 1:], dtype=float),
            "energy_upper": np.asarray(meta["ev_energy_upper"][ev_pos], dtype=float),
            "energy_lower": np.asarray(meta["ev_energy_lower"][ev_pos], dtype=float),
        }
    )
    slot_df["bid_p_delta"] = slot_df["bid_p_rev"] - slot_df["bid_p_day"]
    slot_df["bid_r_delta"] = slot_df["bid_r_rev"] - slot_df["bid_r_day"]
    slot_df["soc_day_start_pct"] = slot_df["energy_day_start"] / capacity * 100.0
    slot_df["soc_day_end_pct"] = slot_df["energy_day_end"] / capacity * 100.0
    slot_df["soc_lower_pct"] = slot_df["energy_lower"] / capacity * 100.0
    slot_df["soc_upper_pct"] = slot_df["energy_upper"] / capacity * 100.0
    slot_df["ev_label"] = meta["ev_labels"][ev_pos]
    slot_df["capacity_mwh"] = capacity
    return slot_df


def build_selected_ev_actual_df(sim: dict[str, Any], ev_number: int) -> pd.DataFrame:
    result = sim["result"]
    meta = sim["resource_meta"]
    nofslots = sim["NOFSLOTS"]

    ev_pos = ev_number - 1
    if ev_pos < 0 or ev_pos >= len(meta["ev_labels"]):
        raise IndexError(f"ev_number={ev_number} 超出范围。")

    ev_idx = int(meta["ev_indices"][ev_pos])
    steps_per_slot = result["P_dis_actual"].shape[1] // nofslots
    n_minutes = result["P_dis_actual"].shape[1]
    capacity = _ev_capacity(meta, ev_pos)

    minute_hours = np.arange(n_minutes, dtype=float) / steps_per_slot
    availability_minute = np.repeat(np.asarray(meta["ev_available"][ev_pos], dtype=bool), steps_per_slot)
    lower_minute = np.repeat(np.asarray(meta["ev_energy_lower"][ev_pos], dtype=float), steps_per_slot) / capacity * 100.0
    upper_minute = np.repeat(np.asarray(meta["ev_energy_upper"][ev_pos], dtype=float), steps_per_slot) / capacity * 100.0
    energy_path = np.asarray(result["E_actual"][ev_idx], dtype=float)

    actual_df = pd.DataFrame(
        {
            "minute_idx": np.arange(n_minutes, dtype=int),
            "hour": minute_hours,
            "slot": np.floor(minute_hours).astype(int) + 1,
            "available": availability_minute,
            "p_dis": np.asarray(result["P_dis_actual"][ev_idx], dtype=float),
            "p_ch": np.asarray(result["P_ch_actual"][ev_idx], dtype=float),
            "energy_start": energy_path[:-1],
            "energy_end": energy_path[1:],
            "soc_start_pct": energy_path[:-1] / capacity * 100.0,
            "soc_end_pct": energy_path[1:] / capacity * 100.0,
            "soc_lower_pct": lower_minute,
            "soc_upper_pct": upper_minute,
        }
    )
    actual_df["p_net"] = actual_df["p_dis"] - actual_df["p_ch"]
    actual_df["ev_label"] = meta["ev_labels"][ev_pos]
    actual_df["capacity_mwh"] = capacity
    return actual_df


def build_analysis_bundle(sim: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary_df": build_summary_df(sim),
        "slot_df": build_slot_df(sim),
        "resource_df": build_resource_df(sim),
        "ev_heatmaps": build_ev_heatmap_data(sim),
    }
