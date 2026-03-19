"""
配置文件 - 集中管理所有硬编码参数
避免在代码中硬编码数值,提高可维护性和可配置性
"""
from dataclasses import dataclass
from typing import List
import numpy as np


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
    """电动汽车配置参数"""
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
    aggregate_evs: bool = True  # 是否按调度计划聚合EV
    max_evs: int = None  # 最大EV数量限制
    pr_ch: float = 0.0  # $/MWh
    # 离开前能量下限调整系数
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
class TCLConfig:
    """温控负载配置参数"""
    NOFTCL: int = 0  # TCL数量 (与MATLAB数据文件保持一致，param_day_21_pv_es_ev.mat中无TCL)
    tcl_c: List[float] = None  # 等效电容 (MWh/K)
    tcl_r: List[float] = None  # 等效电阻 (K/MWh)
    tcl_cop: List[float] = None  # 循环效率
    h_load_ratio: List[float] = None  # 热负载分配比例
    T_ref: float = 28.0  # 参考温度(K)
    T_init: float = 26.0  # 初始温度(K)
    energy_upper_limit: float = 4.0  # MWh
    energy_lower_limit: float = 0.0  # MWh
    power_ch_limit: List[float] = None  # kW
    power_lower_limit: float = 0.0  # kW
    theta_base: float = 1.0

    def __post_init__(self):
        if self.NOFTCL > 0:
            if self.tcl_c is None:
                self.tcl_c = [80.0, 80.0, 40.0][:self.NOFTCL]
            if self.tcl_r is None:
                self.tcl_r = [0.1, 0.1, 0.15][:self.NOFTCL]
            if self.tcl_cop is None:
                self.tcl_cop = [3.6, 3.6, 3.3][:self.NOFTCL]
            if self.h_load_ratio is None:
                self.h_load_ratio = [0.4, 0.4, 0.2][:self.NOFTCL]
            if self.power_ch_limit is None:
                self.power_ch_limit = [400.0, 400.0, 200.0][:self.NOFTCL]
        else:
            # 当NOFTCL=0时，使用空列表
            self.tcl_c = []
            self.tcl_r = []
            self.tcl_cop = []
            self.h_load_ratio = []
            self.power_ch_limit = []

    @property
    def energy_init(self) -> float:
        return self.T_ref - self.T_init

    @property
    def power_ch_upper_limit(self) -> np.ndarray:
        return np.array(self.power_ch_limit) * 1e-3  # MW


@dataclass
class IPPConfig:
    """工业负荷配置参数"""
    NOFIPP: int = 0  # IPP数量 (与MATLAB数据文件保持一致，param_day_21_pv_es_ev.mat中无IPP)
    energy_init_ratio: float = 0.5  # 初始存储比例
    energy_upper_limit_ratio: float = 0.90  # 上限比例
    energy_lower_limit_ratio: float = 0.10  # 下限比例
    power_lower_limit: float = 0.0  # MW
    theta: float = 1.0
    # 最后几个时段能量下限计算的时段数
    final_slots_count: int = 14  # 最后14个时段
    # 瓶颈过程需要工作的时长
    bottleneck_working_hours: float = 22.0  # 小时
    bottleneck_process_index: int = 0  # 瓶颈过程索引(最后一个)

    @property
    def power_ch_lower_limit(self) -> np.ndarray:
        return np.zeros(self.NOFIPP)


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
