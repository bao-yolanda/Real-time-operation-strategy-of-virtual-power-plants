from typing import Any, Literal
import numpy as np
import os
import sys
import argparse

# 获取项目根目录（Real-time-operation-strategy-of-virtual-power-plants-main）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from myexp.max_profit_1 import max_profit_1
from myexp.max_profit_t import max_profit_t
from myexp.fast_control_implement import fast_control_implement
from myexp.data_process.prepare_main import prepare_main_data
from myexp.mat_utils import as_1d, as_2d, load_mat


def _normalize_param(param: Any) -> Any:
    param.price_e = as_1d(param.price_e)
    param.price_reg = as_2d(param.price_reg)
    param.hourly_Mileage = as_1d(param.hourly_Mileage)
    param.hourly_Distribution = as_2d(param.hourly_Distribution)
    param.d_s = as_1d(param.d_s)
    param.index_none_reg = as_1d(param.index_none_reg).astype(int)
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


def main(data_source: Literal["data_prepare", "data_process"] = "data_process",
         mat_file: str = None) -> None:
    """主函数：运行VPP调频优化仿真

    Args:
        data_source: 数据来源
            - "data_prepare": 从MATLAB的data_prepare文件夹读取.mat文件
            - "data_process": 使用Python的data_process模块生成数据
        mat_file: .mat文件路径（当data_source="data_prepare"时使用）
            如果为None，默认读取'data_prepare/param_day_21_pv_es_ev.mat'
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
        config.tcl.NOFTCL = 0  # 不包含TCL
        config.ipp.NOFIPP = 0  # 不包含IPP

        param, param_std, time_params, Signal_day = prepare_main_data(
            day_price=21,
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
    print(f"数据源: {data_source}, 资源组成: PV + ES + EV (无IPP和TCL)")

    NOFTCAP_bid = 900
    NOFTCAP_ctrl = 30

    result = {
        "P_alloc": np.zeros((NOFDER, 0)),
        "actualMil": np.zeros(NOFSLOTS),
        "actualEnergy": np.zeros(NOFSLOTS),
        "actualCost": np.zeros(NOFSLOTS),
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

    max_profit_1(ctx)

    for t_cap in range(1, (NOFSLOTS - 1) * 1800 + 1):
        if t_cap % NOFTCAP_bid == 1:
            delta_t_rest = delta_t - ((t_cap - 1) % 1800) / 1800.0
            ctx.update({"t_cap": t_cap, "delta_t_rest": delta_t_rest})
            max_profit_t(ctx)
        if t_cap % NOFTCAP_ctrl == 1:
            ctx.update({"t_cap": t_cap})
            fast_control_implement(ctx)

    for t_cap in range((NOFSLOTS - 1) * 1800 + 1, NOFSLOTS * 1800):
        if t_cap % NOFTCAP_ctrl == 1:
            ctx.update({"t_cap": t_cap})
            fast_control_implement(ctx)

    result["actualEnegyFee"] = param.price_e * result["actualEnergy"]
    result["actualProfit"] = (
        param.price_reg[:, 0] * result["Bid_R_rev"] * param.s_perf
        + (param.price_reg[:, 1] * result["actualMil"]) * param.s_perf
    )
    result["actualProfit"] = result["actualProfit"] * delta_t

    print("done")
    print(f"profit: {result['actualProfit'].sum():.4f}")


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
        default='data_prepare',
        choices=['data_prepare', 'data_process'],
        help='数据来源: data_prepare(读取MATLAB数据) 或 data_process(生成Python数据)'
    )
    parser.add_argument(
        '--mat_file',
        type=str,
        default=None,
        help='.mat文件路径（仅当data_source=data_prepare时有效）'
    )

    args = parser.parse_args()
    main(data_source=args.data_source, mat_file=args.mat_file)
