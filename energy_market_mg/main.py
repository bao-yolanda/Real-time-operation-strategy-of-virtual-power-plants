from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from energy_market_mg.data_utils import build_case
    from energy_market_mg.dayahead import solve_day_ahead
    from energy_market_mg.intraday import run_intraday_rolling
else:
    from .data_utils import build_case
    from .dayahead import solve_day_ahead
    from .intraday import run_intraday_rolling


def _save_market_power_plot(case, day_ahead, intraday, output_file: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    hours = np.arange(1, case.horizon + 1)
    es_day = day_ahead["es_discharge"] - day_ahead["es_charge"]
    es_rt = intraday["es_discharge"] - intraday["es_charge"]

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    fig.suptitle("Day-Ahead vs Intraday Dispatch Under Forecast Uncertainty", fontsize=15, fontweight="bold")

    ax = axes[0]
    ax.plot(hours, day_ahead["net_grid"], label="Day-ahead net grid", color="#1f3b73", linewidth=2.4)
    ax.plot(hours, intraday["net_grid"], label="Intraday net grid", color="#d95f02", linewidth=2.2)
    ax.fill_between(hours, day_ahead["net_grid"], intraday["net_grid"], color="#f3c7a6", alpha=0.25)
    ax.axhline(0.0, color="#888888", linewidth=0.9)
    ax.set_ylabel("MW")
    ax.set_title("Grid Exchange")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=10)

    ax = axes[1]
    ax.plot(hours, day_ahead["ev_charge"], label="Day-ahead EV charge", color="#2a9d8f", linewidth=2.3)
    ax.plot(hours, intraday["ev_charge"], label="Intraday EV charge", color="#e76f51", linewidth=2.1)
    ax.plot(hours, case.ev_power_floor_dayahead, label="Day-ahead EV floor", color="#2a9d8f", linewidth=1.5, linestyle="--")
    ax.plot(hours, case.ev_power_actual, label="Actual EV availability", color="#e76f51", linewidth=1.5, linestyle="--")
    ax.set_ylabel("MW")
    ax.set_title("EV Charging Response")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=9)

    ax = axes[2]
    ax.plot(hours, case.pv_forecast, label="PV forecast", color="#7cb518", linewidth=2.1)
    ax.plot(hours, case.pv_floor_dayahead, label="Day-ahead PV floor", color="#7cb518", linewidth=1.6, linestyle="--")
    ax.plot(hours, case.pv_actual, label="PV actual", color="#a7c957", linewidth=2.0)
    ax.plot(hours, es_day, label="Day-ahead ES net output", color="#6a4c93", linewidth=1.8, linestyle=":")
    ax.plot(hours, es_rt, label="Intraday ES net output", color="#ff006e", linewidth=1.8, linestyle='-.')
    ax.set_ylabel("MW")
    ax.set_title("PV Uncertainty and Storage Response")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=9)

    ax = axes[3]
    ax.plot(hours, case.load_forecast, label="Load forecast", color="#264653", linewidth=2.1)
    ax.plot(hours, case.load_ceiling_dayahead, label="Day-ahead load ceiling", color="#264653", linewidth=1.6, linestyle="--")
    ax.plot(hours, case.load_actual, label="Load actual", color="#f4a261", linewidth=2.0)
    ax.plot(hours, case.ev_power_floor_dayahead, label="EV floor", color="#577590", linewidth=1.4, linestyle=":")
    ax.plot(hours, case.ev_power_actual, label="EV actual", color="#e76f51", linewidth=1.4, linestyle='-.')
    ax.set_ylabel("MW")
    ax.set_xlabel("Hour")
    ax.set_title("Load and EV Uncertainty")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=9)

    for ax in axes:
        ax.set_xlim(1, case.horizon)
        ax.set_xticks(hours)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def _summarize(case, day_ahead, intraday) -> pd.DataFrame:
    day_cost = float(case.energy_price @ day_ahead["grid_import"] - case.sell_price @ day_ahead["grid_export"])
    rt_cost = float(case.energy_price @ intraday["grid_import"] - case.sell_price @ intraday["grid_export"])

    day_df = np.abs(np.diff(day_ahead["net_grid"], prepend=0.0)).sum()
    rt_df = np.abs(np.diff(intraday["net_grid"], prepend=0.0)).sum()

    rows = [
        ("dayahead_net_energy_cost_usd", day_cost),
        ("intraday_net_energy_cost_usd", rt_cost),
        ("dayahead_pv_curtailment_mwh", float(day_ahead["pv_curt"].sum())),
        ("intraday_pv_curtailment_mwh", float(intraday["pv_curt"].sum())),
        ("dayahead_grid_ramp_mw", float(day_df)),
        ("intraday_grid_ramp_mw", float(rt_df)),
        ("ev_actual_total_demand_mwh", float(case.ev_cum_demand_actual[-1])),
        ("dayahead_ev_planned_charge_mwh", float(day_ahead["ev_charge"].sum() * case.delta_t)),
        ("intraday_ev_realized_charge_mwh", float(intraday["ev_charge"].sum() * case.delta_t)),
        ("intraday_final_es_energy_mwh", float(intraday["es_energy_end"][-1])),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def _hourly_results(case, day_ahead, intraday) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hour": np.arange(1, case.horizon + 1),
            "price_buy": case.energy_price,
            "pv_forecast": case.pv_forecast,
            "pv_actual": case.pv_actual,
            "load_forecast": case.load_forecast,
            "load_actual": case.load_actual,
            "ev_power_floor_dayahead": case.ev_power_floor_dayahead,
            "ev_power_actual": case.ev_power_actual,
            "dayahead_net_grid": day_ahead["net_grid"],
            "intraday_net_grid": intraday["net_grid"],
            "dayahead_ev_charge": day_ahead["ev_charge"],
            "intraday_ev_charge": intraday["ev_charge"],
            "dayahead_pv_curt": day_ahead["pv_curt"],
            "intraday_pv_curt": intraday["pv_curt"],
            "dayahead_es_energy_end": day_ahead["es_energy"][1:],
            "intraday_es_energy_end": intraday["es_energy_end"],
        }
    )


def _case_metadata(case) -> pd.DataFrame:
    rows = [
        ("horizon", case.horizon),
        ("delta_t_h", case.delta_t),
        ("grid_limit_mw", case.grid_limit_mw),
        ("es_energy_init_mwh", case.es_energy_init_mwh),
        ("es_energy_min_mwh", case.es_energy_min_mwh),
        ("es_energy_max_mwh", case.es_energy_max_mwh),
        ("ev_count", case.metadata.get("ev_count")),
        ("ev_forecast_total_demand_mwh", case.metadata.get("ev_forecast_total_demand_mwh")),
        ("ev_actual_total_demand_mwh", case.metadata.get("ev_actual_total_demand_mwh")),
        ("dro_samples", case.metadata.get("dro_samples")),
        ("dro_alpha", case.metadata.get("dro_alpha")),
    ]
    return pd.DataFrame(rows, columns=["field", "value"])


def main(
    ev_count: int = 120,
    seed: int = 42,
    load_scale_mw: float = 1.8,
    grid_limit_mw: float = 3.5,
    save_results: bool = True,
) -> dict[str, pd.DataFrame]:
    case = build_case(
        ev_count=ev_count,
        seed=seed,
        load_scale_mw=load_scale_mw,
        grid_limit_mw=grid_limit_mw,
    )
    day_ahead = solve_day_ahead(case)
    intraday = run_intraday_rolling(case)

    summary = _summarize(case, day_ahead, intraday)
    hourly = _hourly_results(case, day_ahead, intraday)
    metadata = _case_metadata(case)

    print("Energy-only microgrid scheduling finished.")
    print(summary.to_string(index=False))

    if save_results:
        results_dir = Path(__file__).resolve().parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = results_dir / f"energy_market_mg_summary_{stamp}.csv"
        hourly_file = results_dir / f"energy_market_mg_hourly_{stamp}.csv"
        metadata_file = results_dir / f"energy_market_mg_case_{stamp}.csv"
        excel_file = results_dir / f"energy_market_mg_{stamp}.xlsx"
        plot_file = results_dir / f"energy_market_mg_power_curves_{stamp}.png"

        summary.to_csv(summary_file, index=False, encoding="utf-8-sig")
        hourly.to_csv(hourly_file, index=False, encoding="utf-8-sig")
        metadata.to_csv(metadata_file, index=False, encoding="utf-8-sig")
        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            metadata.to_excel(writer, sheet_name="case", index=False)
            summary.to_excel(writer, sheet_name="summary", index=False)
            hourly.to_excel(writer, sheet_name="hourly", index=False)
        plot_saved = _save_market_power_plot(case, day_ahead, intraday, plot_file)

        print(f"Saved summary to: {summary_file}")
        print(f"Saved hourly results to: {hourly_file}")
        print(f"Saved case metadata to: {metadata_file}")
        print(f"Saved workbook to: {excel_file}")
        if plot_saved:
            print(f"Saved power curves to: {plot_file}")
        else:
            print("Skipped power curve plot because matplotlib is not available.")

    return {"case": metadata, "summary": summary, "hourly": hourly}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Energy-only microgrid day-ahead and intraday scheduling.")
    parser.add_argument("--ev_count", type=int, default=40, help="Number of EV charging tasks used in the case.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for actual profiles and EV uncertainty.")
    parser.add_argument("--load_scale_mw", type=float, default=1.8, help="Base load scaling in MW.")
    parser.add_argument("--grid_limit_mw", type=float, default=3.5, help="Grid import/export limit in MW.")
    parser.add_argument("--no_save", action="store_true", help="Disable CSV/XLSX export.")
    args = parser.parse_args()
    main(
        ev_count=args.ev_count,
        seed=args.seed,
        load_scale_mw=args.load_scale_mw,
        grid_limit_mw=args.grid_limit_mw,
        save_results=not args.no_save,
    )

