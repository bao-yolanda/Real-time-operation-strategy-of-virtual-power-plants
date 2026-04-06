from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POINTS_PER_HOUR = 1800
HOURS_PER_DAY = 24

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["pdf.fonttype"] = 42


@dataclass
class RegDStats:
    day_reg: int
    history_days: list[int]
    granularity: float
    scenario_values: np.ndarray
    signal_day: np.ndarray
    historical_distribution: np.ndarray
    actual_distribution: np.ndarray
    historical_mileage: np.ndarray
    actual_mileage: np.ndarray
    selected_hours: list[int]


def _project_root(base_dir: str | Path | None = None) -> Path:
    if base_dir is not None:
        return Path(base_dir).resolve()
    return Path(__file__).resolve().parent.parent


def _load_regd_signals(base_dir: str | Path | None = None) -> np.ndarray:
    root = _project_root(base_dir)
    source_file = root / "data_prepare" / "07 2020.xlsx"
    df = pd.read_excel(source_file, header=None, sheet_name="Dynamic", usecols=range(1, 32))
    signals = (
        df.iloc[1 : 1 + HOURS_PER_DAY * POINTS_PER_HOUR, :]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    return np.clip(signals, -1.0, 1.0)


def _scenario_values(granularity: float) -> np.ndarray:
    mid_points = np.arange(-1.0 + 0.5 * granularity, 1.0, granularity, dtype=float)
    return np.concatenate(([-1.0], mid_points, [1.0]))


def _signal_to_bin_index(signal_value: float, granularity: float, n_intervals: int) -> int:
    if signal_value >= 0:
        if signal_value > 0.9999:
            return n_intervals - 1
        return int(np.ceil(signal_value / granularity) + 1 / granularity)

    if signal_value < -0.9999:
        return 0
    return int(np.floor(signal_value / granularity) + 1 / granularity + 1)


def _hourly_distribution(signal_slice: np.ndarray, granularity: float) -> np.ndarray:
    n_intervals = int(2 / granularity) + 2
    distribution = np.zeros(n_intervals, dtype=float)
    for value in signal_slice:
        distribution[_signal_to_bin_index(float(value), granularity, n_intervals)] += 1.0
    total = distribution.sum()
    if total == 0:
        return np.full(n_intervals, 1.0 / n_intervals, dtype=float)
    return distribution / total


def _hourly_mileage(signal_slice: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(signal_slice))))


def build_regd_stats(
    day_reg: int = 22,
    nof_his_days: int = 14,
    granularity: float = 0.1,
    base_dir: str | Path | None = None,
    top_hours: int = 3,
) -> RegDStats:
    signals = _load_regd_signals(base_dir)
    total_days = signals.shape[1]
    if day_reg < 1 or day_reg > total_days:
        raise ValueError(f"day_reg must be between 1 and {total_days}, got {day_reg}")
    if day_reg - nof_his_days < 1:
        raise ValueError(
            f"day_reg={day_reg} does not leave enough room for {nof_his_days} historical days"
        )

    actual_col = day_reg - 1
    history_cols = list(range(day_reg - nof_his_days - 1, day_reg - 1))

    scenario_values = _scenario_values(granularity)
    n_intervals = len(scenario_values)
    signal_day = signals[:, actual_col]

    historical_distribution = np.zeros((HOURS_PER_DAY, n_intervals), dtype=float)
    actual_distribution = np.zeros((HOURS_PER_DAY, n_intervals), dtype=float)
    historical_mileage = np.zeros(HOURS_PER_DAY, dtype=float)
    actual_mileage = np.zeros(HOURS_PER_DAY, dtype=float)

    for hour_idx in range(HOURS_PER_DAY):
        start = hour_idx * POINTS_PER_HOUR
        end = (hour_idx + 1) * POINTS_PER_HOUR

        hist_dist_by_day = []
        hist_mileage_by_day = []
        for history_col in history_cols:
            signal_slice = signals[start:end, history_col]
            hist_dist_by_day.append(_hourly_distribution(signal_slice, granularity))
            hist_mileage_by_day.append(_hourly_mileage(signal_slice))

        historical_distribution[hour_idx, :] = np.mean(hist_dist_by_day, axis=0)
        historical_mileage[hour_idx] = float(np.mean(hist_mileage_by_day))

        actual_slice = signal_day[start:end]
        actual_distribution[hour_idx, :] = _hourly_distribution(actual_slice, granularity)
        actual_mileage[hour_idx] = _hourly_mileage(actual_slice)

    selected_idx = np.argsort(historical_mileage)[-top_hours:]
    selected_hours = sorted(int(hour + 1) for hour in selected_idx)

    return RegDStats(
        day_reg=day_reg,
        history_days=[idx + 1 for idx in history_cols],
        granularity=granularity,
        scenario_values=scenario_values,
        signal_day=signal_day,
        historical_distribution=historical_distribution,
        actual_distribution=actual_distribution,
        historical_mileage=historical_mileage,
        actual_mileage=actual_mileage,
        selected_hours=selected_hours,
    )


def _ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_actual_signal_figure(stats: RegDStats, sample_step: int = 30):
    x_hours = np.arange(0, len(stats.signal_day), sample_step) / POINTS_PER_HOUR
    y_signal = stats.signal_day[::sample_step]

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(x_hours, y_signal, color="#1f4e79", linewidth=0.8)
    ax.set_xlim(0, HOURS_PER_DAY)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(range(0, HOURS_PER_DAY + 1, 2))
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("RegD信号值")
    ax.grid(True, alpha=0.25)
    return fig


def save_actual_signal_plot(stats: RegDStats, output_dir: str | Path, sample_step: int = 30) -> Path:
    out_dir = _ensure_output_dir(output_dir)
    fig = create_actual_signal_figure(stats, sample_step=sample_step)

    output_path = out_dir / "regd_actual_signal_full_day.pdf"
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_distribution_heatmaps_figure(stats: RegDStats):
    vmax = float(np.max(stats.historical_distribution))

    fig, ax = plt.subplots(figsize=(8.2, 6))

    y_ticks = [
        0,
        len(stats.scenario_values) // 4,
        len(stats.scenario_values) // 2,
        3 * len(stats.scenario_values) // 4,
        len(stats.scenario_values) - 1,
    ]
    y_labels = [f"{stats.scenario_values[idx]:.2f}" for idx in y_ticks]

    ax.imshow(
        stats.historical_distribution.T,
        aspect="auto",
        origin="lower",
        cmap="Blues",
        vmin=0.0,
        vmax=vmax,
        extent=(0.5, HOURS_PER_DAY + 0.5, -0.5, len(stats.scenario_values) - 0.5),
    )
    ax.set_xlabel("小时")
    ax.set_xticks(range(1, HOURS_PER_DAY + 1, 2))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylabel("场景值")
    return fig


def save_distribution_heatmaps(stats: RegDStats, output_dir: str | Path) -> Path:
    out_dir = _ensure_output_dir(output_dir)
    fig = create_distribution_heatmaps_figure(stats)

    output_path = out_dir / "regd_distribution_heatmaps.pdf"
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_hourly_mileage_figure(stats: RegDStats):
    hours = np.arange(1, HOURS_PER_DAY + 1)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(
        hours,
        stats.historical_mileage,
        marker="o",
        linewidth=2.0,
        color="#2a9d8f",
        label="历史平均里程",
    )
    ax.plot(
        hours,
        stats.actual_mileage,
        marker="s",
        linewidth=2.0,
        color="#e76f51",
        label="实际日里程",
    )
    ax.set_xlim(1, HOURS_PER_DAY)
    ax.set_xticks(range(1, HOURS_PER_DAY + 1))
    ax.set_xlabel("小时")
    ax.set_ylabel("里程")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return fig


def save_hourly_mileage_plot(stats: RegDStats, output_dir: str | Path) -> Path:
    out_dir = _ensure_output_dir(output_dir)
    fig = create_hourly_mileage_figure(stats)

    output_path = out_dir / "regd_hourly_mileage_compare.pdf"
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_typical_hour_distribution_figure(stats: RegDStats):
    hours = stats.selected_hours
    fig, axes = plt.subplots(len(hours), 1, figsize=(12, 3.4 * len(hours)), sharex=True)
    if len(hours) == 1:
        axes = [axes]

    x = np.arange(len(stats.scenario_values))
    x_labels = [f"{value:.2f}" for value in stats.scenario_values]

    for ax, hour in zip(axes, hours):
        hist_vals = stats.historical_distribution[hour - 1]
        ax.bar(x, hist_vals, color="#9ecae1", width=0.85, label="统计概率分布")
        ax.set_ylabel("概率")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False, loc="upper right")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(x_labels, rotation=45, ha="right")
    axes[-1].set_xlabel("场景值")
    return fig


def save_typical_hour_distribution_plot(stats: RegDStats, output_dir: str | Path) -> Path:
    out_dir = _ensure_output_dir(output_dir)
    fig = create_typical_hour_distribution_figure(stats)

    output_path = out_dir / "regd_typical_hour_distribution_compare.pdf"
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_mileage_summary(stats: RegDStats, output_dir: str | Path) -> Path:
    out_dir = _ensure_output_dir(output_dir)
    df = pd.DataFrame(
        {
            "小时": np.arange(1, HOURS_PER_DAY + 1),
            "历史平均里程": stats.historical_mileage,
            "实际日里程": stats.actual_mileage,
            "差值": stats.actual_mileage - stats.historical_mileage,
        }
    )
    output_path = out_dir / "regd_hourly_mileage_compare.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def save_all_regd_figures(stats: RegDStats, output_dir: str | Path) -> list[Path]:
    outputs = [
        save_actual_signal_plot(stats, output_dir),
        save_distribution_heatmaps(stats, output_dir),
        save_hourly_mileage_plot(stats, output_dir),
        save_typical_hour_distribution_plot(stats, output_dir),
        save_mileage_summary(stats, output_dir),
    ]
    return outputs


def build_all_regd_figures(stats: RegDStats):
    return [
        ("研究日实际RegD信号", create_actual_signal_figure(stats)),
        ("历史统计场景分布", create_distribution_heatmaps_figure(stats)),
        ("逐小时调频里程对比", create_hourly_mileage_figure(stats)),
        ("典型小时统计场景分布", create_typical_hour_distribution_figure(stats)),
    ]


def summarize_stats(stats: RegDStats) -> pd.DataFrame:
    selected = [hour - 1 for hour in stats.selected_hours]
    return pd.DataFrame(
        {
            "指标": [
                "研究日",
                "历史窗口",
                "离散粒度",
                "典型小时",
                "重点小时历史平均里程",
                "重点小时实际平均里程",
            ],
            "数值": [
                stats.day_reg,
                f"{stats.history_days[0]}-{stats.history_days[-1]}",
                stats.granularity,
                ", ".join(str(hour) for hour in stats.selected_hours),
                float(np.mean(stats.historical_mileage[selected])),
                float(np.mean(stats.actual_mileage[selected])),
            ],
        }
    )


if __name__ == "__main__":
    stats_obj = build_regd_stats()
    export_dir = _project_root() / "results" / "regd_signal_figures"
    saved_files = save_all_regd_figures(stats_obj, export_dir)
    print("已保存文件:")
    for file_path in saved_files:
        print(file_path)
