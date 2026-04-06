"""
资源参数准备
配置光伏、储能、EV、TCL、IPP的参数
支持多类型EV和数量扩展
"""
import numpy as np
import openpyxl
import os
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field

from .config import ResourceConfig, default_config


# ========== EV类型配置 ==========
@dataclass
class EVTypeConfig:
    """单个EV类型的配置参数"""
    battery_capacity: float  # kWh 电池容量
    energy_init_ratio: float  # 初始SOC比例
    energy_end_ratio: float  # 目标SOC比例
    energy_upper_limit_ratio: float = 0.95  # SOC上限比例
    energy_lower_limit_ratio: float = 0.05  # SOC下限比例
    power_ch_limit: float = 22.0  # kW 充电功率
    power_dis_limit: float = 22.0  # kW 放电功率
    eta_ch: float = 0.90  # 充电效率
    eta_dis: float = 0.90  # 放电效率
    pr_ch: float = 0.0  # $/MWh 充电老化成本
    pr_dis: float = 150.0  # $/MWh 放电老化成本

    @property
    def energy_init(self) -> float:
        """初始能量 (MWh)"""
        return self.battery_capacity * self.energy_init_ratio * 1e-3

    @property
    def energy_end(self) -> float:
        """目标能量 (MWh)"""
        return self.battery_capacity * self.energy_end_ratio * 1e-3

    @property
    def energy_upper_limit(self) -> float:
        """能量上限 (MWh)"""
        return self.battery_capacity * self.energy_upper_limit_ratio * 1e-3

    @property
    def energy_lower_limit(self) -> float:
        """能量下限 (MWh)"""
        return self.battery_capacity * self.energy_lower_limit_ratio * 1e-3

    @property
    def power_ch_upper_limit(self) -> float:
        """充电功率上限 (MW)"""
        return self.power_ch_limit * 1e-3

    @property
    def power_dis_upper_limit(self) -> float:
        """放电功率上限 (MW)"""
        return self.power_dis_limit * 1e-3


# 定义三种EV类型 (根据用户需求)
EV_TYPES: Dict[str, EVTypeConfig] = {
    'a': EVTypeConfig(
        battery_capacity=75.0,      # kWh
        energy_init_ratio=0.40,     # 40%
        energy_end_ratio=0.90,      # 90%
        power_ch_limit=11.0,        # kW
        power_dis_limit=11.0,       # kW
        eta_ch=0.90,
        eta_dis=0.90,
        pr_dis=150.0,
        pr_ch=0.0,
    ),
    'b': EVTypeConfig(
        battery_capacity=60.0,      # kWh
        energy_init_ratio=0.333,    # 33.3%
        energy_end_ratio=0.833,     # 83.3%
        power_ch_limit=22.0,        # kW
        power_dis_limit=22.0,       # kW
        eta_ch=0.90,
        eta_dis=0.90,
        pr_dis=150.0,
        pr_ch=0.0,
    ),
    'c': EVTypeConfig(
        battery_capacity=45.0,      # kWh
        energy_init_ratio=0.30,     # 30%
        energy_end_ratio=0.80,      # 80%
        power_ch_limit=7.0,         # kW
        power_dis_limit=7.0,        # kW
        eta_ch=0.90,
        eta_dis=0.90,
        pr_dis=150.0,
        pr_ch=0.0,
    ),
}

# 类型分配比例 (可调整)
EV_TYPE_RATIOS = {'a': 0.33, 'b': 0.34, 'c': 0.33}  # 约1:1:1


# ========== 资源参数类 ==========
@dataclass
class ResourceParameters:
    """资源参数类"""
    # 光伏参数
    power_dis_upper_limit_pv: np.ndarray
    power_dis_lower_limit_pv: np.ndarray

    # 储能参数
    energy_init_es: float
    energy_upper_limit_es: float
    energy_lower_limit_es: float
    power_dis_upper_limit_es: float
    power_dis_lower_limit_es: float
    power_ch_upper_limit_es: float
    power_ch_lower_limit_es: float
    theta_es: float
    eta_dis_es: float
    eta_ch_es: float
    pr_dis_es: float
    pr_ch_es: float

    # EV参数 (支持向量化，每辆EV可以有不同参数)
    energy_init_ev: np.ndarray  # (NOFEV,) 每辆EV的初始能量
    energy_end_ev: np.ndarray  # (NOFEV,) 每辆EV的目标能量
    energy_upper_limit_ev: np.ndarray  # (NOFEV,)
    energy_lower_limit_ev: np.ndarray  # (NOFEV,)
    power_dis_upper_limit_ev: np.ndarray  # (NOFEV,)
    power_dis_lower_limit_ev: np.ndarray  # (NOFEV,)
    power_ch_upper_limit_ev: np.ndarray  # (NOFEV,)
    power_ch_lower_limit_ev: np.ndarray  # (NOFEV,)
    theta_ev: float
    eta_dis_ev: np.ndarray  # (NOFEV,)
    eta_ch_ev: np.ndarray  # (NOFEV,)
    pr_dis_ev: np.ndarray  # (NOFEV,)
    pr_ch_ev: np.ndarray  # (NOFEV,)
    NOFEV: int
    u: np.ndarray  # 充电状态矩阵 (NOFEV, NOFSLOTS)
    ev_types: np.ndarray = None  # (NOFEV,) 每辆EV的类型
    ev_group_counts: np.ndarray = None  # 每组EV的数量 (NOFEV,)，聚合模式下有效
    ev_group_mapping: dict = None  # 组索引 -> 原始EV索引列表

    # 兼容旧代码的标量属性 (聚合后的等效值)
    _energy_init_ev_scalar: float = None
    _energy_end_ev_scalar: float = None
    _energy_upper_limit_ev_scalar: float = None
    _energy_lower_limit_ev_scalar: float = None
    _power_dis_upper_limit_ev_scalar: float = None
    _power_ch_upper_limit_ev_scalar: float = None


def _assign_ev_types(n_evs: int, type_ratios: Dict[str, float] = None, seed: int = 42) -> List[str]:
    """
    为EV分配类型
    
    Args:
        n_evs: EV数量
        type_ratios: 类型比例，如 {'a': 0.33, 'b': 0.34, 'c': 0.33}
        seed: 随机种子
    
    Returns:
        types: 类型列表 ['a', 'b', 'a', 'c', ...]
    """
    if type_ratios is None:
        type_ratios = EV_TYPE_RATIOS
    
    np.random.seed(seed)
    
    # 按比例计算各类型数量
    type_names = list(type_ratios.keys())
    type_probs = [type_ratios[t] for t in type_names]
    
    # 使用多项分布分配类型
    type_indices = np.random.multinomial(n_evs, type_probs)
    
    types = []
    for i, count in enumerate(type_indices):
        types.extend([type_names[i]] * count)
    
    # 打乱顺序
    np.random.shuffle(types)
    
    return types


def _expand_ev_data(EV_data: np.ndarray, target_count: int, seed: int = 42) -> np.ndarray:
    """
    按比例扩展EV数据
    
    Args:
        EV_data: 原始EV数据 (n_original, 3)
        target_count: 目标EV数量
        seed: 随机种子
    
    Returns:
        expanded_data: 扩展后的EV数据 (target_count, 3)
    """
    n_original = len(EV_data)
    
    if target_count <= n_original:
        return EV_data[:target_count]
    
    np.random.seed(seed)
    
    # 计算需要复制的次数
    n_copies = target_count // n_original
    remainder = target_count % n_original
    
    # 完整复制
    expanded = np.tile(EV_data, (n_copies, 1))
    
    # 随机选择补充
    if remainder > 0:
        extra_indices = np.random.choice(n_original, remainder, replace=True)
        expanded = np.vstack([expanded, EV_data[extra_indices]])
    
    # 打乱顺序
    np.random.shuffle(expanded)
    
    return expanded


def _get_ev_params_by_types(ev_types: List[str]) -> Tuple[np.ndarray, ...]:
    """
    根据EV类型列表获取各参数数组
    
    Returns:
        energy_init, energy_end, energy_upper, energy_lower,
        power_dis, power_ch, eta_dis, eta_ch, pr_dis, pr_ch
    """
    n = len(ev_types)
    
    energy_init = np.zeros(n)
    energy_end = np.zeros(n)
    energy_upper = np.zeros(n)
    energy_lower = np.zeros(n)
    power_dis = np.zeros(n)
    power_ch = np.zeros(n)
    eta_dis = np.zeros(n)
    eta_ch = np.zeros(n)
    pr_dis = np.zeros(n)
    pr_ch = np.zeros(n)
    
    for i, t in enumerate(ev_types):
        cfg = EV_TYPES[t]
        energy_init[i] = cfg.energy_init
        energy_end[i] = cfg.energy_end
        energy_upper[i] = cfg.energy_upper_limit
        energy_lower[i] = cfg.energy_lower_limit
        power_dis[i] = cfg.power_dis_upper_limit
        power_ch[i] = cfg.power_ch_upper_limit
        eta_dis[i] = cfg.eta_dis
        eta_ch[i] = cfg.eta_ch
        pr_dis[i] = cfg.pr_dis
        pr_ch[i] = cfg.pr_ch
    
    return (energy_init, energy_end, energy_upper, energy_lower,
            power_dis, power_ch, eta_dis, eta_ch, pr_dis, pr_ch)


def _aggregate_ev_by_schedule_multi_type(
    EV_data: np.ndarray,
    u: np.ndarray,
    ev_types: List[str],
    NOFSLOTS: int,
) -> tuple:
    """
    按(arrive_slot, depart_slot, type)对EV进行聚合
    
    同类型、同一调度计划的EV可以合并
    """
    schedule_groups = {}
    for idx in range(len(EV_data)):
        arrive = int(EV_data[idx, 1])
        depart = int(EV_data[idx, 2])
        ev_type = ev_types[idx]
        key = (arrive, depart, ev_type)
        if key not in schedule_groups:
            schedule_groups[key] = []
        schedule_groups[key].append(idx)

    n_groups = len(schedule_groups)
    u_agg = np.zeros((n_groups, NOFSLOTS))
    group_counts = np.zeros(n_groups, dtype=int)
    group_types = []
    group_mapping = {}

    for group_idx, (key, ev_indices) in enumerate(sorted(schedule_groups.items())):
        arrive, depart, ev_type = key
        u_agg[group_idx, :] = u[ev_indices[0], :]
        group_counts[group_idx] = len(ev_indices)
        group_types.append(ev_type)
        group_mapping[group_idx] = ev_indices

    return u_agg, group_counts, group_types, group_mapping, n_groups


def prepare_parameters(
    NOFSLOTS: int = None,
    base_dir: str = None,
    config: ResourceConfig = None,
    target_ev_count: int = None,  # 目标EV数量，用于敏感性分析
    ev_type_ratios: Dict[str, float] = None,  # EV类型比例
    ev_seed: int = 42,  # 随机种子
) -> Tuple[ResourceParameters, Dict]:
    """
    准备所有资源参数

    Args:
        NOFSLOTS: 时段数量,如果为None则使用配置文件中的默认值
        base_dir: 基础目录
        config: 资源配置对象,如果为None则使用默认配置
        target_ev_count: 目标EV数量，用于敏感性分析 (120, 240, 360, 480等)
        ev_type_ratios: EV类型比例，如 {'a': 0.33, 'b': 0.34, 'c': 0.33}
        ev_seed: 随机种子

    Returns:
        param: 资源参数对象
        param_dict: 参数字典 (兼容旧代码)
    """
    if config is None:
        config = default_config

    if NOFSLOTS is None:
        NOFSLOTS = config.system.NOFSLOTS

    if base_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(os.path.dirname(script_dir))

    # ========== 光伏参数 ==========
    output_pv = prepare_pv_output(base_dir)
    power_dis_upper_limit_pv = output_pv  # (NOFSLOTS,)
    power_dis_lower_limit_pv = np.zeros(NOFSLOTS)

    # ========== 储能参数 ==========
    es_cfg = config.es
    energy_init_es = es_cfg.energy_init  # MWh
    energy_upper_limit_es = es_cfg.energy_upper_limit  # MWh
    energy_lower_limit_es = es_cfg.energy_lower_limit  # MWh
    power_dis_upper_limit_es = es_cfg.power_dis_upper_limit  # MW
    power_dis_lower_limit_es = es_cfg.power_dis_lower_limit
    power_ch_upper_limit_es = es_cfg.power_ch_upper_limit  # MW
    power_ch_lower_limit_es = es_cfg.power_ch_lower_limit
    theta_es = es_cfg.theta
    eta_dis_es = es_cfg.eta_dis
    eta_ch_es = es_cfg.eta_ch
    pr_dis_es = es_cfg.pr_dis  # $/MWh
    pr_ch_es = es_cfg.pr_ch  # $/MWh

    # ========== EV参数 (支持多类型) ==========
    filename = os.path.join(base_dir, 'data_prepare', 'EV_arrive_leave.xlsx')
    wb = openpyxl.load_workbook(filename)
    ws = wb['EV_arrive_leave']

    # 读取EV数据（动态读取所有有效数据行）
    EV_data_original = []
    row_idx = 2  # 从A2开始（第1行是表头）
    while True:
        row_values = []
        has_data = False
        for col_idx in range(3):  # 读取3列
            cell_value = ws.cell(row=row_idx, column=col_idx + 1).value
            if cell_value is not None:
                has_data = True
            row_values.append(cell_value if cell_value is not None else 0.0)
        if not has_data:
            break  # 空行，停止读取
        EV_data_original.append(row_values)
        row_idx += 1
    wb.close()

    EV_data_original = np.array(EV_data_original)
    n_original = len(EV_data_original)
    
    print(f"  原始EV数据: {n_original} 辆")

    # 确定目标EV数量
    ev_cfg = config.ev
    if target_ev_count is not None:
        NOFEV = target_ev_count
    elif hasattr(ev_cfg, 'max_evs') and ev_cfg.max_evs is not None:
        NOFEV = ev_cfg.max_evs
    else:
        NOFEV = n_original

    # 按比例扩展EV数据
    if NOFEV != n_original:
        EV_data = _expand_ev_data(EV_data_original, NOFEV, seed=ev_seed)
        print(f"  EV数量扩展: {n_original} -> {NOFEV}")
    else:
        EV_data = EV_data_original

    # 为每辆EV分配类型
    ev_types = _assign_ev_types(NOFEV, ev_type_ratios, seed=ev_seed)
    ev_types_array = np.array(ev_types)
    
    # 统计类型分布
    from collections import Counter
    type_counts = Counter(ev_types)
    print(f"  EV类型分布: {dict(type_counts)}")

    # 根据类型获取参数
    (energy_init_ev, energy_end_ev, energy_upper_limit_ev, energy_lower_limit_ev,
     power_dis_upper_limit_ev, power_ch_upper_limit_ev, 
     eta_dis_ev, eta_ch_ev, pr_dis_ev, pr_ch_ev) = _get_ev_params_by_types(ev_types)
    
    power_dis_lower_limit_ev = np.zeros(NOFEV)
    power_ch_lower_limit_ev = np.zeros(NOFEV)

    # 充电状态矩阵 u (NOFEV, NOFSLOTS)
    u = np.zeros((NOFEV, NOFSLOTS))
    for idx in range(NOFEV):
        arrive_slot = int(EV_data[idx, 1]) - 1  # 转换为0索引
        depart_slot = int(EV_data[idx, 2]) - 1
        for jdx in range(NOFSLOTS):
            if arrive_slot <= jdx <= depart_slot:
                u[idx, jdx] = 1.0

    print(f"  在场EV数量范围: [{int(u.sum(axis=0).min())}, {int(u.sum(axis=0).max())}]")

    # ========== EV聚合：按(arrive, depart, type)分组 ==========
    ev_group_counts = None
    ev_group_mapping = None

    if hasattr(ev_cfg, 'aggregate_evs') and ev_cfg.aggregate_evs:
        u_agg, ev_group_counts, group_types, ev_group_mapping, n_groups = _aggregate_ev_by_schedule_multi_type(
            EV_data, u, ev_types, NOFSLOTS
        )
        print(f"  EV聚合: {NOFEV}辆 -> {n_groups}组")
        
        # 聚合后更新参数
        (energy_init_ev, energy_end_ev, energy_upper_limit_ev, energy_lower_limit_ev,
         power_dis_upper_limit_ev, power_ch_upper_limit_ev, 
         eta_dis_ev, eta_ch_ev, pr_dis_ev, pr_ch_ev) = _get_ev_params_by_types(group_types)
        
        power_dis_lower_limit_ev = np.zeros(n_groups)
        power_ch_lower_limit_ev = np.zeros(n_groups)
        
        # 聚合后功率按组内车辆数放大
        for gid in range(n_groups):
            count = ev_group_counts[gid]
            power_dis_upper_limit_ev[gid] *= count
            power_ch_upper_limit_ev[gid] *= count
            energy_init_ev[gid] *= count
            energy_end_ev[gid] *= count
            energy_upper_limit_ev[gid] *= count
            energy_lower_limit_ev[gid] *= count
        
        u = u_agg
        ev_types_array = np.array(group_types)
        NOFEV = n_groups

    theta_ev = 1.0

    # ========== 组装参数对象 ==========
    param = ResourceParameters(
        # 光伏
        power_dis_upper_limit_pv=power_dis_upper_limit_pv,
        power_dis_lower_limit_pv=power_dis_lower_limit_pv,

        # 储能
        energy_init_es=energy_init_es,
        energy_upper_limit_es=energy_upper_limit_es,
        energy_lower_limit_es=energy_lower_limit_es,
        power_dis_upper_limit_es=power_dis_upper_limit_es,
        power_dis_lower_limit_es=power_dis_lower_limit_es,
        power_ch_upper_limit_es=power_ch_upper_limit_es,
        power_ch_lower_limit_es=power_ch_lower_limit_es,
        theta_es=theta_es,
        eta_dis_es=eta_dis_es,
        eta_ch_es=eta_ch_es,
        pr_dis_es=pr_dis_es,
        pr_ch_es=pr_ch_es,

        # EV (向量化参数)
        energy_init_ev=energy_init_ev,
        energy_end_ev=energy_end_ev,
        energy_upper_limit_ev=energy_upper_limit_ev,
        energy_lower_limit_ev=energy_lower_limit_ev,
        power_dis_upper_limit_ev=power_dis_upper_limit_ev,
        power_dis_lower_limit_ev=power_dis_lower_limit_ev,
        power_ch_upper_limit_ev=power_ch_upper_limit_ev,
        power_ch_lower_limit_ev=power_ch_lower_limit_ev,
        theta_ev=theta_ev,
        eta_dis_ev=eta_dis_ev,
        eta_ch_ev=eta_ch_ev,
        pr_dis_ev=pr_dis_ev,
        pr_ch_ev=pr_ch_ev,
        NOFEV=NOFEV,
        u=u,
        ev_types=ev_types_array,
        ev_group_counts=ev_group_counts,
        ev_group_mapping=ev_group_mapping,
    )

    # ========== 转换为字典 (兼容旧代码) ==========
    param_dict = {
        'power_dis_upper_limit_pv': power_dis_upper_limit_pv,
        'power_dis_lower_limit_pv': power_dis_lower_limit_pv,
        'energy_init_es': energy_init_es,
        'energy_upper_limit_es': energy_upper_limit_es,
        'energy_lower_limit_es': energy_lower_limit_es,
        'power_dis_upper_limit_es': power_dis_upper_limit_es,
        'power_dis_lower_limit_es': power_dis_lower_limit_es,
        'power_ch_upper_limit_es': power_ch_upper_limit_es,
        'power_ch_lower_limit_es': power_ch_lower_limit_es,
        'theta_es': theta_es,
        'eta_dis_es': eta_dis_es,
        'eta_ch_es': eta_ch_es,
        'pr_dis_es': pr_dis_es,
        'pr_ch_es': pr_ch_es,
        # EV参数 (向量化)
        'energy_init_ev': energy_init_ev,
        'energy_end_ev': energy_end_ev,
        'energy_upper_limit_ev': energy_upper_limit_ev,
        'energy_lower_limit_ev': energy_lower_limit_ev,
        'power_dis_upper_limit_ev': power_dis_upper_limit_ev,
        'power_dis_lower_limit_ev': power_dis_lower_limit_ev,
        'power_ch_upper_limit_ev': power_ch_upper_limit_ev,
        'power_ch_lower_limit_ev': power_ch_lower_limit_ev,
        'theta_ev': theta_ev,
        'eta_dis_ev': eta_dis_ev,
        'eta_ch_ev': eta_ch_ev,
        'pr_dis_ev': pr_dis_ev,
        'pr_ch_ev': pr_ch_ev,
        'NOFEV': NOFEV,
        'u': u,
        'ev_types': ev_types_array,
        'ev_group_counts': ev_group_counts,
        'ev_group_mapping': ev_group_mapping,
    }

    return param, param_dict


def prepare_pv_output(base_dir: str = None) -> np.ndarray:
    """内部函数：读取光伏出力"""
    if base_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(os.path.dirname(script_dir))

    import scipy.io as sio
    filename = os.path.join(base_dir, 'data_prepare', 'output_pv.mat')
    data = sio.loadmat(filename)
    output_pv = np.asarray(data['output_pv']).flatten()

    return output_pv


def print_ev_type_info():
    """打印EV类型信息"""
    print("=" * 80)
    print("EV类型参数配置")
    print("=" * 80)
    
    for type_name, cfg in EV_TYPES.items():
        print(f"\n类型 {type_name}:")
        print(f"  电池容量: {cfg.battery_capacity} kWh")
        print(f"  初始SOC: {cfg.energy_init_ratio*100:.1f}% ({cfg.battery_capacity * cfg.energy_init_ratio:.1f} kWh)")
        print(f"  目标SOC: {cfg.energy_end_ratio*100:.1f}% ({cfg.battery_capacity * cfg.energy_end_ratio:.1f} kWh)")
        print(f"  SOC范围: [{cfg.energy_lower_limit_ratio*100:.0f}%, {cfg.energy_upper_limit_ratio*100:.0f}%]")
        print(f"  充电功率: {cfg.power_ch_limit} kW")
        print(f"  放电功率: {cfg.power_dis_limit} kW")
        print(f"  需充电量: {(cfg.energy_end - cfg.energy_init)*1000:.1f} kWh")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print_ev_type_info()
    print("\n测试不同EV数量:")
    
    for target in [120, 240, 360, 480]:
        print(f"\n--- 目标EV数量: {target} ---")
        param, param_dict = prepare_parameters(target_ev_count=target)
        print(f"  实际EV数量: {param.NOFEV}")
        print(f"  能量初始化范围: [{param.energy_init_ev.min():.4f}, {param.energy_init_ev.max():.4f}] MWh")
        print(f"  充电功率范围: [{param.power_ch_upper_limit_ev.min():.4f}, {param.power_ch_upper_limit_ev.max():.4f}] MW")
