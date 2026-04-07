from typing import Any, Literal, Optional
import argparse
import os
import sys
from datetime import datetime

import numpy as np

# 获取项目根目录（Real-time-operation-strategy-of-virtual-power-plants-main）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from myexp.fast_control_implement import fast_control_implement
from myexp.mat_utils import as_1d, as_2d, load_mat
from myexp.max_profit_1_admm import max_profit_1_admm
from myexp.max_profit_t_admm import max_profit_t_admm
from myexp.data_process.prepare_main import prepare_main_data


def _normalize_param(param: Any) -> Any:
    param.price_e = as_1d(param.price_e)
    param.price_reg = as_2d(param.price_reg)
    param.hourly_Mileage = as_1d(param.hourly_Mileage)
    param.hourly_Distribution = as_2d(param.hourly_Distribution)
    param.d_s = as_1d(param.d_s)
    param.s_perf = float(param.s_perf)
    return param


def _normalize_param_std(param_std: Any) -> Any:
    param_std.energy_init = as_1d(param_std.energy_init)
    param_std.energy_upper_limit = as_2d(param_std.energy_upper_limit)
    param_std.energy_lower_limit = as_2d(param_std.energy_lower_limit)
    param_std.power_dis_upper_limit = as_2d(param_std.power_dis_upper_limit)
    param_std.power_dis_lower_limit = as_2d(param_std.power_dis_lower_limit)
    param_std.power_ch_upper_limit = as_2d(param_std.power_ch_upper_limit)
    param_std.power_ch_lower_limit = as_2d(param_std.power_ch_lower_limit)
    param_std.theta = as_1d(param_std.theta)
    param_std.eta_ch = as_2d(param_std.eta_ch)
    param_std.eta_dis = as_2d(param_std.eta_dis)
    param_std.pr_dis = as_1d(param_std.pr_dis)
    param_std.pr_ch = as_1d(param_std.pr_ch)
    param_std.wOmiga = as_2d(param_std.wOmiga)
    return param_std


def _safe_percentage_change(current: float, baseline: float) -> float:
    if abs(baseline) < 1e-9:
        return 0.0
    return (current - baseline) / baseline * 100.0


def main(
    data_source: Literal["data_prepare", "data_process"] = "data_process",
    mat_file: str = None,
    save_results: bool = True,
    run_full_simulation: bool = False,
    aggregate_evs: bool = False,
    parallel_user_solve: bool = True,
    user_solver_workers: int = 0,
    ev_count: Optional[int] = None,
    admm_max_iter: int = 10,
    rho_p: float = 5.0,
    rho_r: float = 5.0,
    abs_tol: float = 1e-3,
    rel_tol: float = 1e-3,
    admm_verbose: bool = False,
    progress_every: int = 5,
) -> None:
    """运行 VPP 调频优化的 ADMM 版本。"""
    if data_source == "data_prepare":
        if mat_file is None:
            mat_file = os.path.join("data_prepare", "param.mat")

        print(f"从data_prepare读取数据: {mat_file}")
        data = load_mat(mat_file).raw

        param = _normalize_param(data["param"])
        param_std = _normalize_param_std(data["param_std"])

        Signal_day = as_1d(data["Signal_day"])
        NOFSLOTS = int(np.asarray(data["NOFSLOTS"]).squeeze())
        NOFDER = int(np.asarray(data["NOFDER"]).squeeze())
        NOFSCEN = int(np.asarray(data["NOFSCEN"]).squeeze())
        delta_t = float(np.asarray(data["delta_t"]).squeeze())
        M = float(np.asarray(data.get("M", 1e6)).squeeze())
        delta_t_req = float(np.asarray(data.get("delta_t_req", 0.5)).squeeze())

    else:
        print("使用data_process模块生成数据（只包含PV、ES、EV）")

        from myexp.data_process.config import ResourceConfig

        config = ResourceConfig()
        config.ev.aggregate_evs = aggregate_evs
        if ev_count is not None:
            config.ev.max_evs = ev_count

        param, param_std, time_params, Signal_day = prepare_main_data(
            day_price=25,
            hour_init=0,
            NOFSLOTS=24,
            granularity=0.1,
            nofHisDays=14,
            M=1e6,
            delta_t_req=0.5,
            s_perf=0.984,
            config=config,
        )

        param = _normalize_param(param)
        param_std = _normalize_param_std(param_std)

        NOFSLOTS = time_params["NOFSLOTS"]
        NOFDER = time_params["NOFDER"]
        NOFSCEN = time_params["NOFSCEN"]
        delta_t = time_params["delta_t"]
        M = time_params["M"]
        delta_t_req = time_params["delta_t_req"]

    print(f"NOFSLOTS={NOFSLOTS}, NOFDER={NOFDER}, NOFSCEN={NOFSCEN}")
    print(f"delta_t={delta_t}, delta_t_req={delta_t_req}, M={M}")
    print(f"数据源: {data_source}, 资源组成: PV + ES + EV")

    NOFTCAP_bid = 1200
    NOFTCAP_ctrl = 30

    result = {
        "Bid_P_day": np.zeros(NOFSLOTS),
        "Bid_R_day": np.zeros(NOFSLOTS),
        "P_DER_day": np.zeros((NOFDER, NOFSLOTS)),
        "R_DER_day": np.zeros((NOFDER, NOFSLOTS)),
        "E_day": np.zeros((NOFDER, NOFSLOTS + 1)),
        "Bid_P_rev": np.zeros(NOFSLOTS),
        "Bid_R_rev": np.zeros(NOFSLOTS),
        "P_DER_rev": np.zeros((NOFDER, NOFSLOTS)),
        "R_DER_rev": np.zeros((NOFDER, NOFSLOTS)),
        "E_rev": np.zeros((NOFDER, NOFSLOTS + 1)),
        "revision_times": np.zeros(NOFSLOTS),
        "P_alloc": np.zeros((NOFDER, 0)),
        "Signal_actual": np.zeros(NOFSLOTS * 1800 // NOFTCAP_ctrl),
        "P_dis_actual": np.zeros((NOFDER, NOFSLOTS * 1800 // NOFTCAP_ctrl)),
        "P_ch_actual": np.zeros((NOFDER, NOFSLOTS * 1800 // NOFTCAP_ctrl)),
        "E_actual": np.zeros((NOFDER, NOFSLOTS * 1800 // NOFTCAP_ctrl + 1)),
        "actualMil": np.zeros(NOFSLOTS),
        "actualEnergy": np.zeros(NOFSLOTS),
        "actualCost": np.zeros(NOFSLOTS),
        "EnergyRevenue_day": np.zeros(NOFSLOTS),
        "CapacityRevenue_day": np.zeros(NOFSLOTS),
        "BatteryDeg_day": np.zeros(NOFSLOTS),
        "EnegyFee_day": np.zeros(NOFSLOTS),
        "RegCapacity_day": np.zeros(NOFSLOTS),
        "RegMileage_day": np.zeros(NOFSLOTS),
        "Profit_day": np.zeros(NOFSLOTS),
        "Profit_rev": np.zeros(NOFSLOTS),
        "Profit_realtime": np.zeros(NOFSLOTS),
        "actualEnergyRevenue": np.zeros(NOFSLOTS),
        "actualCapacityRevenue": np.zeros(NOFSLOTS),
        "actualBatteryDeg": np.zeros(NOFSLOTS),
        "actualEnegyFee": np.zeros(NOFSLOTS),
        "actualRegCapacity": np.zeros(NOFSLOTS),
        "actualRegMileage": np.zeros(NOFSLOTS),
        "actualProfit": np.zeros(NOFSLOTS),
    }

    ctx = {
        "param": param,
        "param_std": param_std,
        "Signal_day": Signal_day,
        "NOFSLOTS": NOFSLOTS,
        "NOFDER": NOFDER,
        "NOFSCEN": NOFSCEN,
        "delta_t": delta_t,
        "delta_t_req": delta_t_req,
        "M": M,
        "NOFTCAP_ctrl": NOFTCAP_ctrl,
        "result": result,
        "admm_settings": {
            "max_iter": admm_max_iter,
            "rho_p": rho_p,
            "rho_r": rho_r,
            "abs_tol": abs_tol,
            "rel_tol": rel_tol,
            "verbose": admm_verbose,
            "parallel_user_solve": parallel_user_solve,
            "user_solver_workers": user_solver_workers,
            "progress_every": progress_every,
        },
    }

    print(
        "ADMM设置: "
        "EV块模式=fleet矩阵, "
        f"EV聚合={'开' if data_source == 'data_process' and aggregate_evs else '关/沿用输入数据'}, "
        f"并行用户求解={'开' if parallel_user_solve else '关'}, "
        f"worker={user_solver_workers if user_solver_workers > 0 else 'auto'}, "
        f"max_iter={admm_max_iter}, rho_p={rho_p}, rho_r={rho_r}, "
        f"run_full_simulation={'开' if run_full_simulation else '关'}"
    )
    if data_source == "data_process":
        actual_ev_blocks = NOFDER - 2
        print(f"实际参与优化的EV块数量: {actual_ev_blocks}")

    max_profit_1_admm(ctx)

    result["Bid_P_day"] = result.get("Bid_P_init", np.zeros(NOFSLOTS)).copy()
    result["Bid_R_day"] = result.get("Bid_R_init", np.zeros(NOFSLOTS)).copy()
    result["P_DER_day"] = result.get("P_DER_rev", np.zeros((NOFDER, NOFSLOTS))).copy()
    result["R_DER_day"] = result.get("R_DER_rev", np.zeros((NOFDER, NOFSLOTS))).copy()
    result["E_day"] = result.get("E_init", np.zeros((NOFDER, NOFSLOTS + 1))).copy()

    reg_energy_coeff = (param.hourly_Distribution @ param.d_s) * param.price_e
    result["EnegyFee_day"] = param.price_e * result["Bid_P_day"] * delta_t
    result["RegCapacity_day"] = param.price_reg[:, 0] * result["Bid_R_day"] * param.s_perf * delta_t
    result["RegMileage_day"] = param.price_reg[:, 1] * param.hourly_Mileage * result["Bid_R_day"] * param.s_perf * delta_t
    result["EnergyRevenue_day"] = (param.price_e * result["Bid_P_day"] + reg_energy_coeff * result["Bid_R_day"]) * delta_t
    result["CapacityRevenue_day"] = result["RegCapacity_day"] + result["RegMileage_day"]
    result["BatteryDeg_day"] = result.get("admm_deg_day", np.zeros(NOFSLOTS)).copy()
    result["Profit_day"] = result["EnergyRevenue_day"] + result["CapacityRevenue_day"] - result["BatteryDeg_day"]

    print(f"日前计划总利润: {result['Profit_day'].sum():.4f} USD")
    print(f"  - 能量收益: {result['EnergyRevenue_day'].sum():.4f} USD")
    print(f"  - 容量收益: {result['CapacityRevenue_day'].sum():.4f} USD")
    print(f"  - 电池退化费用: {result['BatteryDeg_day'].sum():.4f} USD")
    print(f"  - EV子块平均计算时间: {result.get('admm_user_solve_time_avg', 0.0):.6f} s")
    print(f"  - 主块平均计算时间: {result.get('admm_system_time_avg', 0.0):.6f} s")

    if not run_full_simulation:
        print("\n已默认跳过滚动修正和实时控制。需要完整仿真时请加 `--run_full_simulation`。", flush=True)
        print("\ndone")
        return

    for t_cap in range(1, (NOFSLOTS - 1) * 1800 + 1):
        if t_cap % NOFTCAP_bid == 1:
            delta_t_rest = delta_t - ((t_cap - 1) % 1800) / 1800.0
            ctx.update({"t_cap": t_cap, "delta_t_rest": delta_t_rest})
            max_profit_t_admm(ctx)
        if t_cap % NOFTCAP_ctrl == 1:
            ctx.update({"t_cap": t_cap})
            fast_control_implement(ctx)

    for t_cap in range((NOFSLOTS - 1) * 1800 + 1, NOFSLOTS * 1800):
        if t_cap % NOFTCAP_ctrl == 1:
            ctx.update({"t_cap": t_cap})
            fast_control_implement(ctx)

    result["actualEnegyFee"] = param.price_e * result["actualEnergy"]
    result["actualRegCapacity"] = param.price_reg[:, 0] * result["Bid_R_rev"] * param.s_perf * delta_t
    result["actualRegMileage"] = (param.price_reg[:, 1] * result["actualMil"]) * param.s_perf * delta_t
    result["actualEnergyRevenue"] = result["actualEnegyFee"].copy()
    result["actualCapacityRevenue"] = result["actualRegCapacity"] + result["actualRegMileage"]
    result["actualBatteryDeg"] = result["actualCost"].copy()
    result["actualProfit"] = result["actualEnergyRevenue"] + result["actualCapacityRevenue"] - result["actualBatteryDeg"]

    print("\n========== 仿真结果汇总 ==========")
    print(f"日前计划总利润: {result['Profit_day'].sum():.4f} USD")
    print(f"  - 能量收益: {result['EnergyRevenue_day'].sum():.4f} USD")
    print(f"  - 容量收益: {result['CapacityRevenue_day'].sum():.4f} USD")
    print(f"  - 电池退化费用: {result['BatteryDeg_day'].sum():.4f} USD")

    print(f"\n实际总利润: {result['actualProfit'].sum():.4f} USD")
    print(f"  - 能量收益: {result['actualEnergyRevenue'].sum():.4f} USD")
    print(f"  - 容量收益: {result['actualCapacityRevenue'].sum():.4f} USD")
    print(f"  - 电池退化费用: {result['actualBatteryDeg'].sum():.4f} USD")

    day_profit_total = float(result["Profit_day"].sum())
    actual_profit_total = float(result["actualProfit"].sum())
    profit_diff = actual_profit_total - day_profit_total
    profit_change_pct = _safe_percentage_change(actual_profit_total, day_profit_total)
    print(f"\n利润变化: {profit_diff:+.4f} USD ({profit_change_pct:+.2f}%)")

    if save_results:
        import pandas as pd

        results_dir = os.path.join(project_root, "results")
        os.makedirs(results_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        df_results = pd.DataFrame({
            "时段": range(1, NOFSLOTS + 1),
            "日前基础功率": result["Bid_P_day"],
            "日前调频功率": result["Bid_R_day"],
            "修正后基础功率": result["Bid_P_rev"],
            "修正后调频功率": result["Bid_R_rev"],
            "日前总收益": result["Profit_day"],
            "实际总收益": result["actualProfit"],
            "日前能量收益": result["EnergyRevenue_day"],
            "日前容量收益": result["CapacityRevenue_day"],
            "日前电池退化费用": result["BatteryDeg_day"],
            "实际能量收益": result["actualEnergyRevenue"],
            "实际容量收益": result["actualCapacityRevenue"],
            "实际电池退化费用": result["actualBatteryDeg"],
            "实际里程": result["actualMil"],
            "实际能量": result["actualEnergy"],
            "实际成本": result["actualCost"],
            "修正次数": result["revision_times"],
        })

        csv_file = os.path.join(results_dir, f"vpp_results_admm_{timestamp}.csv")
        df_results.to_csv(csv_file, index=False, encoding="utf-8-sig")
        print(f"\n主结果CSV已保存到: {csv_file}")

        resource_summary_list = []
        for t in range(NOFSLOTS):
            resource_summary_list.append({
                "时段": t + 1,
                "资源类型": "PV",
                "日前基础功率": result["P_DER_day"][0, t],
                "日前调频功率": result["R_DER_day"][0, t],
                "修正后基础功率": result["P_DER_rev"][0, t],
                "修正后调频功率": result["R_DER_rev"][0, t],
            })
            resource_summary_list.append({
                "时段": t + 1,
                "资源类型": "ES",
                "日前基础功率": result["P_DER_day"][1, t],
                "日前调频功率": result["R_DER_day"][1, t],
                "修正后基础功率": result["P_DER_rev"][1, t],
                "修正后调频功率": result["R_DER_rev"][1, t],
            })

            ev_base_day = result["P_DER_day"][2:, t].sum()
            ev_reg_day = result["R_DER_day"][2:, t].sum()
            ev_base_rev = result["P_DER_rev"][2:, t].sum()
            ev_reg_rev = result["R_DER_rev"][2:, t].sum()
            resource_summary_list.append({
                "时段": t + 1,
                "资源类型": "EV",
                "日前基础功率": ev_base_day,
                "日前调频功率": ev_reg_day,
                "修正后基础功率": ev_base_rev,
                "修正后调频功率": ev_reg_rev,
            })

        df_resource = pd.DataFrame(resource_summary_list)
        csv_resource = os.path.join(results_dir, f"vpp_resource_summary_admm_{timestamp}.csv")
        df_resource.to_csv(csv_resource, index=False, encoding="utf-8-sig")
        print(f"资源汇总CSV已保存到: {csv_resource}")

        df_market = pd.DataFrame({
            "时段": range(1, NOFSLOTS + 1),
            "能量价格": param.price_e,
            "调频容量价格": param.price_reg[:, 0],
            "调频里程价格": param.price_reg[:, 1],
            "小时里程": param.hourly_Mileage,
        })
        csv_market = os.path.join(results_dir, f"vpp_market_data_admm_{timestamp}.csv")
        df_market.to_csv(csv_market, index=False, encoding="utf-8-sig")
        print(f"市场参数CSV已保存到: {csv_market}")

        excel_file = os.path.join(results_dir, f"vpp_results_admm_{timestamp}.xlsx")
        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            df_results.to_excel(writer, sheet_name="主结果", index=False)
            df_resource.to_excel(writer, sheet_name="资源汇总", index=False)
            df_market.to_excel(writer, sheet_name="市场参数", index=False)
            df_summary = pd.DataFrame({
                "指标": [
                    "日前计划总收益",
                    "实际总收益",
                    "总收益差异",
                    "总收益变化率(%)",
                    "日前能量收益",
                    "日前容量收益",
                    "日前电池退化费用",
                    "实际能量收益",
                    "实际容量收益",
                    "实际电池退化费用",
                    "修正总次数",
                    "修正时段数",
                    "平均基础功率(MW)",
                    "平均调频功率(MW)",
                    "平均实际里程(MW)",
                    "EV数量",
                    "ADMM迭代次数",
                    "ADMM是否收敛",
                ],
                "数值": [
                    day_profit_total,
                    actual_profit_total,
                    profit_diff,
                    profit_change_pct,
                    result["EnergyRevenue_day"].sum(),
                    result["CapacityRevenue_day"].sum(),
                    result["BatteryDeg_day"].sum(),
                    result["actualEnergyRevenue"].sum(),
                    result["actualCapacityRevenue"].sum(),
                    result["actualBatteryDeg"].sum(),
                    result["revision_times"].sum(),
                    np.sum(result["revision_times"] > 0),
                    result["Bid_P_rev"].mean(),
                    result["Bid_R_rev"].mean(),
                    result["actualMil"].mean(),
                    NOFDER - 2,
                    result.get("admm_iterations", 0),
                    int(bool(result.get("admm_converged", False))),
                ],
            })
            df_summary.to_excel(writer, sheet_name="汇总统计", index=False)

            df_profit_composition = pd.DataFrame({
                "项目": ["能量收益", "容量收益", "电池退化费用", "总收益"],
                "日前计划": [
                    result["EnergyRevenue_day"].sum(),
                    result["CapacityRevenue_day"].sum(),
                    result["BatteryDeg_day"].sum(),
                    result["Profit_day"].sum(),
                ],
                "实际": [
                    result["actualEnergyRevenue"].sum(),
                    result["actualCapacityRevenue"].sum(),
                    result["actualBatteryDeg"].sum(),
                    result["actualProfit"].sum(),
                ],
                "差异": [
                    result["actualEnergyRevenue"].sum() - result["EnergyRevenue_day"].sum(),
                    result["actualCapacityRevenue"].sum() - result["CapacityRevenue_day"].sum(),
                    result["actualBatteryDeg"].sum() - result["BatteryDeg_day"].sum(),
                    result["actualProfit"].sum() - result["Profit_day"].sum(),
                ],
            })
            df_profit_composition.to_excel(writer, sheet_name="利润构成对比", index=False)

        print(f"完整Excel文件已保存到: {excel_file}")

    print("\ndone")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VPP调频优化仿真（ADMM版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python myexp/main_admm.py
  python myexp/main_admm.py --no_save
  python myexp/main_admm.py --aggregate_evs --ev_count 360
  python myexp/main_admm.py --run_full_simulation --admm_max_iter 3
        """,
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="data_process",
        choices=["data_prepare", "data_process"],
        help="数据来源: data_prepare(读取MATLAB数据) 或 data_process(生成Python数据)",
    )
    parser.add_argument(
        "--mat_file",
        type=str,
        default=None,
        help=".mat文件路径（仅当data_source=data_prepare时有效）",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="不保存结果文件",
    )
    parser.add_argument(
        "--run_full_simulation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否继续执行滚动修正和实时控制（默认关闭，只跑日前ADMM）",
    )
    parser.add_argument(
        "--aggregate_evs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否对EV按到离站时段和类型进行聚合（默认关闭，直接保留车辆矩阵）",
    )
    parser.add_argument(
        "--parallel_user_solve",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否并行求解ADMM中的用户子问题（fleet模式下保留兼容）",
    )
    parser.add_argument(
        "--user_solver_workers",
        type=int,
        default=0,
        help="ADMM用户子问题并行worker数量，0表示自动",
    )
    parser.add_argument(
        "--ev_count",
        type=int,
        default=None,
        help="data_process模式下的EV数量；开启聚合时表示聚合前总EV数",
    )
    parser.add_argument(
        "--admm_max_iter",
        type=int,
        default=10,
        help="ADMM最大迭代次数",
    )
    parser.add_argument(
        "--rho_p",
        type=float,
        default=5.0,
        help="基础功率共识项的罚参数",
    )
    parser.add_argument(
        "--rho_r",
        type=float,
        default=5.0,
        help="调频功率共识项的罚参数",
    )
    parser.add_argument(
        "--abs_tol",
        type=float,
        default=1e-3,
        help="ADMM绝对收敛阈值",
    )
    parser.add_argument(
        "--rel_tol",
        type=float,
        default=1e-3,
        help="ADMM相对收敛阈值",
    )
    parser.add_argument(
        "--admm_verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否输出CVXPY求解细节",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=5,
        help="每隔多少次ADMM迭代打印一次进度",
    )

    args = parser.parse_args()
    main(
        data_source=args.data_source,
        mat_file=args.mat_file,
        save_results=not args.no_save,
        run_full_simulation=args.run_full_simulation,
        aggregate_evs=args.aggregate_evs,
        parallel_user_solve=args.parallel_user_solve,
        user_solver_workers=args.user_solver_workers,
        ev_count=args.ev_count,
        admm_max_iter=args.admm_max_iter,
        rho_p=args.rho_p,
        rho_r=args.rho_r,
        abs_tol=args.abs_tol,
        rel_tol=args.rel_tol,
        admm_verbose=args.admm_verbose,
        progress_every=args.progress_every,
    )
