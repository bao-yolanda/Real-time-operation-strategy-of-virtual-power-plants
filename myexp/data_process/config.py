"""
配置文件 - 集中管理所有硬编码参数
避免在代码中硬编码数值,提高可维护性和可配置性
"""
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np


def _default_type_ratios() -> Dict[str, float]:
    """默认EV类型比例"""
    return {'a': 0.33, 'b': 0.34, 'c': 0.33}


@dataclass
class PVConfig:
    """光伏配置参数"""
    NOFPV: int = 1  # PV数量
    energy_init: float = 1.0  # PV初始能量(没有存储,设为常数)
    energy_end: float = 0.0  # PV结束能量
    energy_upper_limit: float = 2.0  # PV能量上限(常数)
    energy_lower_limit: float = 0.0  # PV能量下限


@dataclass
class ESConfig:
    """储能系统配置参数"""
    energy_init: float = 1.0  # MWh
    energy_upper_limit_ratio: float = 0.9  # 上限比例
    energy_capacity: float = 2.0  # MWh, 总容量
    energy_lower_limit_ratio: float = 0.1  # 下限比例
    power_dis_upper_limit: float = 1.0  # MW
    power_dis_lower_limit: float = 0.0  # MW
    power_ch_upper_limit: float = 1.0  # MW
    power_ch_lower_limit: float = 0.0  # MW
    theta: float = 1.0
    eta_dis: float = 0.90
    eta_ch: float = 0.90
    pr_dis: float = 100.0  # $/MWh
    pr_ch: float = 100.0  # $/MWh

    @property
    def energy_upper_limit(self) -> float:
        return self.energy_upper_limit_ratio * self.energy_capacity

    @property
    def energy_lower_limit(self) -> float:
        return self.energy_lower_limit_ratio * self.energy_capacity


@dataclass
class EVConfig:
    """电动汽车配置参数 - 支持多类型EV"""
    # 多类型EV配置 (在 prepare_parameters.py 中定义具体类型)
    # 类型: a (75kWh, 40%->90%, 11kW), b (60kWh, 33.3%->83.3%, 22kW), c (45kWh, 30%->80%, 7kW)
    
    # 类型分配比例 (可调整) - 使用 default_factory 避免可变默认值问题
    type_ratios: Dict[str, float] = field(default_factory=_default_type_ratios)
    
    # 敏感性分析用：目标EV数量
    target_ev_count: int = None  # 120, 240, 360, 480 等
    
    # 聚合选项
    aggregate_evs: bool = False  # 不聚合，保留每辆EV独立参数
    max_evs: int = None  # 最大EV数量限制 (兼容旧参数)
    
    # 随机种子
    seed: int = 42
    
    # 默认参数 (用于单类型模式，保持向后兼容)
    battery_capacity: float = 60.0  # kWh
    energy_init_ratio: float = 20.0 / 60.0  # 初始SOC
    energy_end_ratio: float = 50.0 / 60.0  # 结束SOC
    energy_upper_limit_ratio: float = 0.9  # 上限比例
    energy_lower_limit_ratio: float = 0.1  # 下限比例
    power_dis_limit: float = 22.0  # kW
    power_ch_limit: float = 22.0  # kW
    power_lower_limit: float = 0.0  # kW
    theta: float = 1.0
    eta_dis: float = 0.90
    eta_ch: float = 0.90
    pr_dis: float = 150.0  # $/MWh
    pr_ch: float = 0.0  # $/MWh
    pre_departure_energy_factor: float = 0.9

    @property
    def energy_init(self) -> float:
        return self.battery_capacity * self.energy_init_ratio * 1e-3  # MWh

    @property
    def energy_end(self) -> float:
        return self.battery_capacity * self.energy_end_ratio * 1e-3  # MWh

    @property
    def energy_upper_limit(self) -> float:
        return self.battery_capacity * self.energy_upper_limit_ratio * 1e-3  # MWh

    @property
    def energy_lower_limit(self) -> float:
        return self.battery_capacity * self.energy_lower_limit_ratio * 1e-3  # MWh

    @property
    def power_dis_upper_limit(self) -> float:
        return self.power_dis_limit * 1e-3  # MW

    @property
    def power_ch_upper_limit(self) -> float:
        return self.power_ch_limit * 1e-3  # MW


@dataclass
class SystemConfig:
    """系统配置参数"""
    NOFSLOTS: int = 24  # 时段数量


@dataclass
class ResourceConfig:
    """资源配置汇总"""
    pv: PVConfig = None
    es: ESConfig = None
    ev: EVConfig = None
    system: SystemConfig = None

    def __post_init__(self):
        if self.pv is None:
            self.pv = PVConfig()
        if self.es is None:
            self.es = ESConfig()
        if self.ev is None:
            self.ev = EVConfig()
        if self.system is None:
            self.system = SystemConfig()


# 默认配置实例
default_config = ResourceConfig()
