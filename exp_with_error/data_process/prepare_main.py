"""
主数据准备模块
整合所有子模块，生成完整的参数文件
"""
import numpy as np
import scipy.io as sio
import os
from typing import Dict, Tuple, Any
from types import SimpleNamespace

from .prepare_wt import prepare_wind_as_regulation
from .prepare_price import prepare_price_data
from .prepare_parameters import prepare_parameters, ResourceParameters
from .prepare_std import prepare_std_parameters
from .config import ResourceConfig, default_config


def dict_to_sns(d: Dict[str, Any]) -> SimpleNamespace:
    """将字典转换为 SimpleNamespace 对象"""
    return SimpleNamespace(**d)


def prepare_main_data(
    day_price: int = 21,
    hour_init: int = 0,
    NOFSLOTS: int = None,
    granularity: float = 0.1,
    nofHisDays: int = 14,
    N_wind: int = 2,
    N_samples: int = 2000,
    M: float = 1e6,
    delta_t_req: float = 0.5,
    s_perf: float = 0.984,
    base_dir: str = None,
    config: ResourceConfig = None
) -> Tuple[SimpleNamespace, SimpleNamespace, Dict[str, Any], np.ndarray]:
    """
    主数据准备函数

    Args:
        day_price: 价格日期
        hour_init: 起始小时
        NOFSLOTS: 时段数量,如果为None则使用配置文件中的默认值
        granularity: 调频信号离散粒度
        nofHisDays: 历史天数
        N_wind: 风电场数量
        N_samples: 风电场景数量
        M: 大M常数（软约束惩罚）
        delta_t_req: 维护时间
        s_perf: 调频性能系数
        base_dir: 基础目录
        config: 资源配置对象,如果为None则使用默认配置

    Returns:
        param: 市场参数（价格、信号分布等）
        param_std: 资源标准化参数
    """
    if config is None:
        config = default_config

    if NOFSLOTS is None:
        NOFSLOTS = config.system.NOFSLOTS

    if base_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(os.path.dirname(script_dir))

    print("="*70)
    print("VPP能量管理 - 数据准备（风电版本）")
    print("="*70)
    print(f"\n配置:")
    print(f"  价格日期: day {day_price}")
    print(f"  风电数据日期: day {day_price + 1}")
    print(f"  时段数(NOFSLOTS): {NOFSLOTS}")
    print(f"  时段长度(delta_t): 1 小时")
    print(f"  风电场数量(N_wind): {N_wind}")
    print(f"  场景数(N_samples): {N_samples}")

    # ========== 步骤1: 读取风电数据并作为调频信号 ==========
    print("\n" + "-"*70)
    print("步骤1: 处理风电数据（风电误差作为调频信号）...")
    print("-"*70)

    day_wind = day_price + 1
    (WT_pred, WT_error_scenarios, WT_full_scenarios,
     hourly_Distribution, hourly_Mileage, d_s, Signal_day) = prepare_wind_as_regulation(
        day_wind=day_wind,
        NOFSLOTS=NOFSLOTS,
        N_wind=N_wind,
        N_samples=N_samples,
        base_dir=base_dir
    )

    NOFSCEN = len(d_s)  # 场景数
    NOFWT = N_wind  # 风电场数量

    # ========== 步骤2: 读取市场价格 ==========
    print("\n" + "-"*70)
    print("步骤2: 读取市场价格数据...")
    print("-"*70)

    price_reg, price_e, start_row = prepare_price_data(
        day_price=day_price,
        hour_init=hour_init,
        NOFSLOTS=NOFSLOTS,
        base_dir=base_dir
    )

    # ========== 步骤3: 准备资源参数 ==========
    print("\n" + "-"*70)
    print("步骤3: 准备资源参数...")
    print("-"*70)

    param, param_dict = prepare_parameters(NOFSLOTS=NOFSLOTS, base_dir=base_dir, config=config)

    # ========== 步骤4: 标准化资源参数 ==========
    print("\n" + "-"*70)
    print("步骤4: 标准化资源参数...")
    print("-"*70)

    param_std = prepare_std_parameters(param, NOFSLOTS=NOFSLOTS, config=config)

    # ========== 组装市场参数 param ==========
    # 计算不参与调频的资源索引
    # 资源索引分配（从 0 开始）：
    # PV: [0, NOFPV-1]
    # ES: [NOFPV, NOFPV]
    # EV: [NOFPV+1, NOFPV+NOFEV]

    # 从配置读取PV数量
    NOFPV = config.pv.NOFPV
    index_none_reg = param_std['index_none_reg']

    param_market = {
        'price_e': price_e,
        'price_reg': price_reg,
        'hourly_Mileage': hourly_Mileage,
        'hourly_Distribution': hourly_Distribution,
        'd_s': d_s,
        'index_none_reg': index_none_reg,
        's_perf': s_perf,

        # 资源信息
        'resource_names': ['pv', 'es', 'ev', 'tcl', 'ipp'],
    }

    NOFDER = NOFPV + 1 + param.NOFEV 

    # 时间参数
    time_params = {
        'NOFSLOTS': NOFSLOTS,
        'NOFSCEN': NOFSCEN,
        'NOFDER': NOFDER,
        'delta_t': 1.0,  # 1小时
        'delta_t_req': delta_t_req,
        'M': M,
    }

    # ========== 步骤5: 合并所有参数 ==========
    print("\n" + "-"*70)
    print("步骤5: 合并参数...")
    print("-"*70)

    print(f"\n  资源总数(NOFDER): {NOFDER}")
    print(f"  光伏(PV): {config.pv.NOFPV}")
    print(f"  储能(ES): 1")
    print(f"  电动汽车(EV): {config.ev.NOFEV if hasattr(config.ev, 'NOFEV') else param.NOFEV}")

    print(f"\n  场景数(NOFSCEN): {NOFSCEN}")
    print(f"  d_s 范围: [{d_s.min():.4f}, {d_s.max():.4f}]")

    # ========== 返回参数 ==========
    # 将字典转换为 SimpleNamespace 以兼容现有代码
    param_market_sns = dict_to_sns(param_market)
    param_std_sns = dict_to_sns(param_std)

    # param: 市场参数
    # param_std: 资源标准参数
    # time_params: 时间参数

    return param_market_sns, param_std_sns, time_params, Signal_day


def save_parameters(
    param_market: SimpleNamespace,
    param_std: SimpleNamespace,
    time_params: Dict[str, Any],
    output_path: str = None,
    day_price: int = 21
) -> None:
    """
    保存参数到 .mat 文件

    Args:
        param_market: 市场参数
        param_std: 资源标准参数
        time_params: 时间参数
        output_path: 输出路径
        day_price: 日期编号
    """
    if output_path is None:
        output_path = f"param_day_{day_price}.mat"

    # 合并所有数据
    # 将 SimpleNamespace 转换为字典以便保存
    mat_data = {
        'param': vars(param_market),
        'param_std': vars(param_std),
        **time_params,
    }

    sio.savemat(output_path, mat_data)
    print(f"\n数据已保存到: {output_path}")
    print(f"  文件大小: {os.path.getsize(output_path) / 1024:.1f} KB")


if __name__ == "__main__":
    # 测试：生成第21天的数据
    param_market, param_std, time_params, Signal_day = prepare_main_data(day_price=21)

    print("\n" + "="*70)
    print("数据准备完成!")
    print("="*70)

    # 保存
    save_parameters(
        param_market=param_market,
        param_std=param_std,
        time_params=time_params,
        day_price=21
    )

    # 打印部分参数验证
    print(f"\n参数验证:")
    print(f"  price_e shape: {param_market['price_e'].shape}")
    print(f"  price_reg shape: {param_market['price_reg'].shape}")
    print(f"  hourly_Distribution shape: {param_market['hourly_Distribution'].shape}")
    print(f"  d_s shape: {param_market['d_s'].shape}")
    print(f"  Signal_day shape: {Signal_day.shape}")
    print(f"  energy_init shape: {param_std['energy_init'].shape}")
    print(f"  energy_upper_limit shape: {param_std['energy_upper_limit'].shape}")
    print(f"  power_dis_upper_limit shape: {param_std['power_dis_upper_limit'].shape}")
