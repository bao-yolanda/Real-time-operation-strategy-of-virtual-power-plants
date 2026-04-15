from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

try:
    from IPython import get_ipython
except ImportError:
    get_ipython = None

if get_ipython is None or get_ipython() is None:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import openpyxl

from myexp.data_process.prepare_price import prepare_price_data


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
class MarketPriceSeries:
    day_price: int
    market_date: str
    hours: np.ndarray
    price_energy: np.ndarray
    price_capacity: np.ndarray
    price_mileage: np.ndarray


def _project_root(base_dir: str | Path | None = None) -> Path:
    if base_dir is not None:
        return Path(base_dir).resolve()
    return Path(__file__).resolve().parent.parent


def _format_market_date(raw_value: object) -> str:
    if isinstance(raw_value, datetime):
        return raw_value.strftime("%Y-%m-%d")
    if raw_value is None:
        return "unknown-date"
    text = str(raw_value).strip()
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return text.split()[0]


def load_day_market_prices(
    day_price: int = 21,
    hour_init: int = 0,
    nofslots: int = 24,
    base_dir: str | Path | None = None,
) -> MarketPriceSeries:
    root = _project_root(base_dir)
    price_reg, price_e, start_row = prepare_price_data(
        day_price=day_price,
        hour_init=hour_init,
        NOFSLOTS=nofslots,
        base_dir=str(root),
    )

    workbook = openpyxl.load_workbook(
        root / "data_prepare" / "regulation_market_results.xlsx",
        data_only=True,
    )
    worksheet = workbook["regulation_market_results"]
    market_date = _format_market_date(worksheet.cell(row=start_row, column=2).value)
    workbook.close()

    return MarketPriceSeries(
        day_price=day_price,
        market_date=market_date,
        hours=np.arange(1, nofslots + 1),
        price_energy=price_e,
        price_capacity=price_reg[:, 0],
        price_mileage=price_reg[:, 1],
    )


def create_market_price_figure(series: MarketPriceSeries):
    fig, ax_left = plt.subplots(figsize=(9.2, 5.2))

    line_energy = ax_left.plot(
        series.hours,
        series.price_energy,
        color="#6F8FA6",
        marker="o",
        linewidth=1.8,
        markersize=3.8,
        label="能量价格 ($/MWh)",
    )[0]

    ax_left.set_xlabel("小时")
    ax_left.set_ylabel("能量价格 ($/MWh)")
    ax_left.set_xticks(range(1, len(series.hours) + 1, 2))
    ax_left.set_xlim(1, len(series.hours))
    ax_left.grid(True, axis="y", linestyle="--", alpha=0.18, linewidth=0.8)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_left.spines["left"].set_color("#9AA6B2")
    ax_left.spines["bottom"].set_color("#9AA6B2")
    ax_left.tick_params(colors="#5B6570")

    ax_right = ax_left.twinx()
    line_capacity = ax_right.plot(
        series.hours,
        series.price_capacity,
        color="#C08A74",
        marker="s",
        linewidth=1.8,
        markersize=3.6,
        label="调频容量价格 ($/MW)",
    )[0]
    line_mileage = ax_right.plot(
        series.hours,
        series.price_mileage,
        color="#86A58D",
        marker="^",
        linewidth=1.8,
        markersize=3.8,
        label="调频里程价格 ($/MW)",
    )[0]
    ax_right.set_ylabel("调频价格 ($/MW)")
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["right"].set_color("#9AA6B2")
    ax_right.tick_params(colors="#5B6570")

    handles = [line_energy, line_capacity, line_mileage]
    labels = [handle.get_label() for handle in handles]
    ax_left.legend(handles, labels, loc="upper left", frameon=False, fontsize=10)

    fig.tight_layout()
    return fig


def save_market_price_figure(
    day_price: int = 21,
    output_dir: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> list[Path]:
    series = load_day_market_prices(day_price=day_price, base_dir=base_dir)
    fig = create_market_price_figure(series)

    if output_dir is None:
        out_dir = _project_root(base_dir) / "results" / "market_price_figures"
    else:
        out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"market_prices_{series.market_date}"
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [pdf_path, png_path]


if __name__ == "__main__":
    saved_files = save_market_price_figure(day_price=21)
    print("已保存文件:")
    for path in saved_files:
        print(path)
