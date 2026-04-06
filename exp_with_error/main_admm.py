from typing import Any, Literal
import numpy as np
import os
import sys
import argparse
from datetime import datetime

# 获取项目根目录（Real-time-operation-strategy-of-virtual-power-plants-main）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from exp_with_error.max_profit_1_admm import max_profit_1_admm
from exp_with_error.max_profit_t_admm import max_profit_t_admm
from exp_with_error.fast_control_implement import fast_control_implement
from myexp.data_process.prepare_main import prepare_main_data
from exp_with_error.mat_utils import as_1d, as_2d, load_mat

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


def main(data_source: Literal["data_prepare", "data_process"] = "data_process",
         mat_file: str = None,
         save_results: bool = True) -> None:
    """主函数：运行VPP调频优化仿真

    Args:
        data_source: 数据来源
            - "data_prepare": 从MATLAB的data_prepare文件夹读取.mat文件
            - "data_process": 使用Python的data_process模块生成数据
        mat_file: .mat文件路径（当data_source="data_prepare"时使用）
            如果为None，默认读取'data_prepare/param_day_21_pv_es_ev.mat'
        save_results: 是否保存详细结果到文件
    """
    if data_source == "data_prepare":
        if mat_file is None:
            mat_file = os.path.join('data_prepare', 'param.mat')

        print(f"从data_prepare读取数据: {mat_file}")
        data = load_mat(mat_file).raw

        param = _normalize_param(data["param"])
        param_std = _normalize_param_std(data["param_std"])

        # 提取其他参数
        Signal_day = as_1d(data["Signal_day"])
        NOFSLOTS = int(np.asarray(data["NOFSLOTS"]).squeeze())
        NOFDER = int(np.asarray(data["NOFDER"]).squeeze())
        NOFSCEN = int(np.asarray(data["NOFSCEN"]).squeeze())
        delta_t = float(np.asarray(data["delta_t"]).squeeze())
        M = float(np.asarray(data.get("M", 1e6)).squeeze())
        delta_t_req = float(np.asarray(data.get("delta_t_req", 0.5)).squeeze())

    else:  # data_source == "data_process"
        print("使用data_process模块生成数据（只包含PV、ES、EV）")

        from myexp.data_process.config import ResourceConfig
        config = ResourceConfig()

        param, param_std, time_params, Signal_day = prepare_main_data(
            day_price=23,
            hour_init=0,
            NOFSLOTS=24,
            granularity=0.1,
            nofHisDays=14,
            M=1e6,
            delta_t_req=0.5,
            s_perf=0.984,
            config=config  # 传入自定义配置
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

    NOFTCAP_bid = 900
    NOFTCAP_ctrl = 30

    # ========== 初始化结果字典 ==========
    # 阶段1: 日前计划 (max_profit_1)
    # 阶段2: 日内修正 (max_profit_t)
    # 阶段3: 实时分配 (fast_control_implement)
    result = {
        # === 日前计划阶段 ===
        "Bid_P_day": np.zeros(NOFSLOTS),      # 日前基础功率投标
        "Bid_R_day": np.zeros(NOFSLOTS),      # 日前调频功率投标
        "P_DER_day": np.zeros((NOFDER, NOFSLOTS)),  # 日前各资源基础功率分配
        "R_DER_day": np.zeros((NOFDER, NOFSLOTS)),  # 日前各资源调频功率分配
        "E_day": np.zeros((NOFDER, NOFSLOTS + 1)),   # 日前能量状态轨迹

        # === 日内修正阶段 ===
        "Bid_P_rev": np.zeros(NOFSLOTS),      # 基础功率投标修正历史
        "Bid_R_rev": np.zeros(NOFSLOTS),      # 调频功率投标修正历史
        "P_DER_rev": np.zeros((NOFDER, NOFSLOTS)),  # 基础功率分配修正历史
        "R_DER_rev": np.zeros((NOFDER, NOFSLOTS)),  # 调频功率分配修正历史
        "E_rev": np.zeros((NOFDER, NOFSLOTS + 1)),   # 能量状态修正历史
        "revision_times": np.zeros(NOFSLOTS),  # 每个时段的修正次数

        # === 实时分配阶段 ===
        "P_alloc": np.zeros((NOFDER, 0)),     # 实时功率分配记录
        "Signal_actual": np.zeros(NOFSLOTS * 1800 // NOFTCAP_ctrl),  # 实际调频信号
        "P_dis_actual": np.zeros((NOFDER, NOFSLOTS * 1800 // NOFTCAP_ctrl)),  # 实际放电功率
        "P_ch_actual": np.zeros((NOFDER, NOFSLOTS * 1800 // NOFTCAP_ctrl)),   # 实际充电功率
        "E_actual": np.zeros((NOFDER, NOFSLOTS * 1800 // NOFTCAP_ctrl + 1)),  # 实际能量状态

        # === 功率平衡数据 ===
        "actualMil": np.zeros(NOFSLOTS),      # 实际里程
        "actualEnergy": np.zeros(NOFSLOTS),   # 实际能量消耗/产出
        "actualCost": np.zeros(NOFSLOTS),     # 实际成本

        # === 利润数据 ===
        "EnergyRevenue_day": np.zeros(NOFSLOTS),   # 日前能量收益（含调频能量项）
        "CapacityRevenue_day": np.zeros(NOFSLOTS), # 日前容量类收益（容量+里程）
        "BatteryDeg_day": np.zeros(NOFSLOTS),      # 日前电池退化费用
        "EnegyFee_day": np.zeros(NOFSLOTS),   # 日前能量费用
        "RegCapacity_day": np.zeros(NOFSLOTS),  # 日前调频容量收益
        "RegMileage_day": np.zeros(NOFSLOTS),   # 日前调频里程收益
        "Profit_day": np.zeros(NOFSLOTS),       # 日前利润

        "Profit_rev": np.zeros(NOFSLOTS),       # 修正后利润
        "Profit_realtime": np.zeros(NOFSLOTS),  # 实时利润

        "actualEnergyRevenue": np.zeros(NOFSLOTS),   # 实际能量收益
        "actualCapacityRevenue": np.zeros(NOFSLOTS), # 实际容量类收益（容量+里程）
        "actualBatteryDeg": np.zeros(NOFSLOTS),      # 实际电池退化费用
        "actualEnegyFee": np.zeros(NOFSLOTS),   # 实际能量费用
        "actualRegCapacity": np.zeros(NOFSLOTS),  # 实际调频容量收益
        "actualRegMileage": np.zeros(NOFSLOTS),   # 实际调频里程收益
        "actualProfit": np.zeros(NOFSLOTS),     # 实际总利润
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
    }

    max_profit_1_admm(ctx)

    # 保存日前计划结果
    result["Bid_P_day"] = result.get("Bid_P_init", np.zeros(NOFSLOTS)).copy()
    result["Bid_R_day"] = result.get("Bid_R_init", np.zeros(NOFSLOTS)).copy()
    result["P_DER_day"] = result.get("P_DER_rev", np.zeros((NOFDER, NOFSLOTS))).copy()
    result["R_DER_day"] = result.get("R_DER_rev", np.zeros((NOFDER, NOFSLOTS))).copy()
    result["E_day"] = result.get("E_init", np.zeros((NOFDER, NOFSLOTS + 1))).copy()

    # 计算日前计划收益：总收益 = 能量收益 + 容量类收益 - 电池退化费用
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
    print(f"  - 用户平均计算时间: {result.get('admm_user_solve_time_avg', 0.0):.6f} s")
    print(f"  - 主站平均计算时间: {result.get('admm_system_time_avg', 0.0):.6f} s")

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

    # ========== 计算各阶段利润 ==========

    # 实际收益计算：总收益 = 能量收益 + 容量类收益 - 电池退化费用
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

    # 利润变化
    day_profit_total = float(result['Profit_day'].sum())
    actual_profit_total = float(result['actualProfit'].sum())
    profit_diff = actual_profit_total - day_profit_total
    profit_change_pct = _safe_percentage_change(actual_profit_total, day_profit_total)
    print(f"\n利润变化: {profit_diff:+.4f} USD ({profit_change_pct:+.2f}%)")

    # ========== 保存结果为CSV和Excel格式 ==========
    if save_results:
        import pandas as pd

        # 创建结果保存目录
        results_dir = os.path.join(project_root, 'results')
        os.makedirs(results_dir, exist_ok=True)

        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ========== 1. 保存主结果CSV (小时级数据) ==========
        df_results = pd.DataFrame({
            '时段': range(1, NOFSLOTS + 1),
            # 日前投标
            '日前基础功率': result['Bid_P_day'],
            '日前调频功率': result['Bid_R_day'],
            # 修正后投标
            '修正后基础功率': result['Bid_P_rev'],
            '修正后调频功率': result['Bid_R_rev'],
            # 利润数据
            '日前总收益': result['Profit_day'],
            '实际总收益': result['actualProfit'],
            # 利润构成
            '日前能量收益': result['EnergyRevenue_day'],
            '日前容量收益': result['CapacityRevenue_day'],
            '日前电池退化费用': result['BatteryDeg_day'],
            '实际能量收益': result['actualEnergyRevenue'],
            '实际容量收益': result['actualCapacityRevenue'],
            '实际电池退化费用': result['actualBatteryDeg'],
            # 功率平衡
            '实际里程': result['actualMil'],
            '实际能量': result['actualEnergy'],
            '实际成本': result['actualCost'],
            # 修正统计
            '修正次数': result['revision_times'],
        })

        csv_file = os.path.join(results_dir, f'vpp_results_{timestamp}.csv')
        df_results.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"\n主结果CSV已保存到: {csv_file}")

        # ========== 2. 保存资源类型汇总CSV (按PV/ES/EV分类) ==========
        # 资源索引: 0=PV, 1=ES, 2~NOFDER-1=EVs
        NOFEV = NOFDER - 2  # EV数量

        # 按资源类型汇总
        resource_summary_list = []
        for t in range(NOFSLOTS):
            # PV (资源0)
            resource_summary_list.append({
                '时段': t + 1,
                '资源类型': 'PV',
                '日前基础功率': result['P_DER_day'][0, t],
                '日前调频功率': result['R_DER_day'][0, t],
                '修正后基础功率': result['P_DER_rev'][0, t],
                '修正后调频功率': result['R_DER_rev'][0, t],
            })

            # ES (资源1)
            resource_summary_list.append({
                '时段': t + 1,
                '资源类型': 'ES',
                '日前基础功率': result['P_DER_day'][1, t],
                '日前调频功率': result['R_DER_day'][1, t],
                '修正后基础功率': result['P_DER_rev'][1, t],
                '修正后调频功率': result['R_DER_rev'][1, t],
            })

            # EV (资源2~NOFDER-1) - 汇总所有EV
            ev_base_day = result['P_DER_day'][2:, t].sum()
            ev_reg_day = result['R_DER_day'][2:, t].sum()
            ev_base_rev = result['P_DER_rev'][2:, t].sum()
            ev_reg_rev = result['R_DER_rev'][2:, t].sum()

            resource_summary_list.append({
                '时段': t + 1,
                '资源类型': 'EV',
                '日前基础功率': ev_base_day,
                '日前调频功率': ev_reg_day,
                '修正后基础功率': ev_base_rev,
                '修正后调频功率': ev_reg_rev,
            })

        df_resource = pd.DataFrame(resource_summary_list)
        csv_resource = os.path.join(results_dir, f'vpp_resource_summary_{timestamp}.csv')
        df_resource.to_csv(csv_resource, index=False, encoding='utf-8-sig')
        print(f"资源汇总CSV已保存到: {csv_resource}")

        # ========== 3. 保存市场参数CSV ==========
        df_market = pd.DataFrame({
            '时段': range(1, NOFSLOTS + 1),
            '能量价格': param.price_e,
            '调频容量价格': param.price_reg[:, 0],
            '调频里程价格': param.price_reg[:, 1],
            '小时里程': param.hourly_Mileage,
        })

        csv_market = os.path.join(results_dir, f'vpp_market_data_{timestamp}.csv')
        df_market.to_csv(csv_market, index=False, encoding='utf-8-sig')
        print(f"市场参数CSV已保存到: {csv_market}")

        # ========== 4. 保存为Excel格式 (多工作表) ==========
        excel_file = os.path.join(results_dir, f'vpp_results_{timestamp}.xlsx')

        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # 工作表1: 主结果
            df_results.to_excel(writer, sheet_name='主结果', index=False)

            # 工作表2: 资源汇总 (按PV/ES/EV分类)
            df_resource.to_excel(writer, sheet_name='资源汇总', index=False)

            # 工作表3: 市场参数
            df_market.to_excel(writer, sheet_name='市场参数', index=False)

            # 工作表4: 汇总统计
            df_summary = pd.DataFrame({
                '指标': [
                    '日前计划总收益',
                    '实际总收益',
                    '总收益差异',
                    '总收益变化率(%)',
                    '日前能量收益',
                    '日前容量收益',
                    '日前电池退化费用',
                    '实际能量收益',
                    '实际容量收益',
                    '实际电池退化费用',
                    '修正总次数',
                    '修正时段数',
                    '平均基础功率(MW)',
                    '平均调频功率(MW)',
                    '平均实际里程(MW)',
                    'EV数量',
                ],
                '数值': [
                    day_profit_total,
                    actual_profit_total,
                    profit_diff,
                    profit_change_pct,
                    result['EnergyRevenue_day'].sum(),
                    result['CapacityRevenue_day'].sum(),
                    result['BatteryDeg_day'].sum(),
                    result['actualEnergyRevenue'].sum(),
                    result['actualCapacityRevenue'].sum(),
                    result['actualBatteryDeg'].sum(),
                    result['revision_times'].sum(),
                    np.sum(result['revision_times'] > 0),
                    result['Bid_P_rev'].mean(),
                    result['Bid_R_rev'].mean(),
                    result['actualMil'].mean(),
                    NOFDER - 2,  # EV数量
                ],
            })
            df_summary.to_excel(writer, sheet_name='汇总统计', index=False)

            # 工作表5: 利润构成对比
            df_profit_composition = pd.DataFrame({
                '项目': ['能量收益', '容量收益', '电池退化费用', '总收益'],
                '日前计划': [
                    result['EnergyRevenue_day'].sum(),
                    result['CapacityRevenue_day'].sum(),
                    result['BatteryDeg_day'].sum(),
                    result['Profit_day'].sum(),
                ],
                '实际': [
                    result['actualEnergyRevenue'].sum(),
                    result['actualCapacityRevenue'].sum(),
                    result['actualBatteryDeg'].sum(),
                    result['actualProfit'].sum(),
                ],
                '差异': [
                    result['actualEnergyRevenue'].sum() - result['EnergyRevenue_day'].sum(),
                    result['actualCapacityRevenue'].sum() - result['CapacityRevenue_day'].sum(),
                    result['actualBatteryDeg'].sum() - result['BatteryDeg_day'].sum(),
                    result['actualProfit'].sum() - result['Profit_day'].sum(),
                ],
            })
            df_profit_composition.to_excel(writer, sheet_name='利润构成对比', index=False)

        print(f"完整Excel文件已保存到: {excel_file}")
        print(f"\n数据概览:")
        print(f"  - 主结果: {len(df_results)} 时段数据")
        print(f"  - 资源汇总: {len(df_resource)} 条记录 (PV/ES/EV × {NOFSLOTS}时段)")
        print(f"  - 市场参数: {len(df_market)} 时段数据")
        print(f"  - Excel文件包含 5 个工作表")

    print("\ndone")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VPP调频优化仿真",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用data_process模块生成数据（默认）
  python myexp/main.py

  # 从data_prepare读取MATLAB数据
  python myexp/main.py --data_source data_prepare

  # 指定.mat文件路径
  python myexp/main.py --data_source data_prepare --mat_file data_prepare/param_day_21.mat
        """
    )
    parser.add_argument(
        '--data_source',
        type=str,
        default='data_process',
        choices=['data_prepare', 'data_process'],
        help='数据来源: data_prepare(读取MATLAB数据) 或 data_process(生成Python数据)'
    )
    parser.add_argument(
        '--mat_file',
        type=str,
        default=None,
        help='.mat文件路径（仅当data_source=data_prepare时有效）'
    )
    parser.add_argument(
        '--no_save',
        action='store_true',
        help='不保存结果文件'
    )

    args = parser.parse_args()
    main(data_source=args.data_source, mat_file=args.mat_file, save_results=not args.no_save)
