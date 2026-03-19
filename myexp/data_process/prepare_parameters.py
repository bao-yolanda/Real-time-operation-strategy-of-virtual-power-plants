"""
资源参数准备
配置光伏、储能、EV、TCL、IPP的参数
"""
import numpy as np
import openpyxl
import os
from typing import Dict, Tuple
from dataclasses import dataclass

from .config import ResourceConfig, default_config


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

    # EV参数
    energy_init_ev: float
    energy_end_ev: float
    energy_upper_limit_ev: float
    energy_lower_limit_ev: float
    power_dis_upper_limit_ev: float
    power_dis_lower_limit_ev: float
    power_ch_upper_limit_ev: float
    power_ch_lower_limit_ev: float
    theta_ev: float
    eta_dis_ev: float
    eta_ch_ev: float
    pr_dis_ev: float
    pr_ch_ev: float
    NOFEV: int
    u: np.ndarray  # 充电状态矩阵 (NOFEV, NOFSLOTS)
    ev_group_counts: np.ndarray = None  # 每组EV的数量 (NOFEV,)，聚合模式下有效
    ev_group_mapping: dict = None  # 组索引 -> 原始EV索引列表


def _aggregate_ev_by_schedule(
    EV_data: np.ndarray,
    u: np.ndarray,
    NOFSLOTS: int,
) -> tuple:
    """
    按(arrive_slot, depart_slot)对EV进行聚合。

    同一调度计划的EV具有相同的约束结构，可以合并为一个"super-EV"，
    其功率和容量按组内车辆数等比例放大。

    Returns:
        u_agg: 聚合后的u矩阵 (n_groups, NOFSLOTS)
        group_counts: 每组的EV数量 (n_groups,)
        group_mapping: {group_idx: [原始EV索引列表]}
        n_groups: 组数
    """
    schedule_groups = {}
    for idx in range(len(EV_data)):
        arrive = int(EV_data[idx, 1])
        depart = int(EV_data[idx, 2])
        key = (arrive, depart)
        if key not in schedule_groups:
            schedule_groups[key] = []
        schedule_groups[key].append(idx)

    n_groups = len(schedule_groups)
    u_agg = np.zeros((n_groups, NOFSLOTS))
    group_counts = np.zeros(n_groups, dtype=int)
    group_mapping = {}

    for group_idx, (key, ev_indices) in enumerate(sorted(schedule_groups.items())):
        u_agg[group_idx, :] = u[ev_indices[0], :]
        group_counts[group_idx] = len(ev_indices)
        group_mapping[group_idx] = ev_indices

    return u_agg, group_counts, group_mapping, n_groups


def prepare_parameters(
    NOFSLOTS: int = None,
    base_dir: str = None,
    config: ResourceConfig = None
) -> Tuple[ResourceParameters, Dict]:
    """
    准备所有资源参数

    Args:
        NOFSLOTS: 时段数量,如果为None则使用配置文件中的默认值
        base_dir: 基础目录
        config: 资源配置对象,如果为None则使用默认配置

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

    # ========== EV参数 ==========
    filename = os.path.join(base_dir, 'data_prepare', 'EV_arrive_leave.xlsx')
    wb = openpyxl.load_workbook(filename)
    ws = wb['EV_arrive_leave']

    # 读取EV数据（动态读取所有有效数据行）
    EV_data = []
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
        EV_data.append(row_values)
        row_idx += 1
    wb.close()

    EV_data = np.array(EV_data)
    # 从配置读取EV参数
    ev_cfg = config.ev

    # 应用EV数量限制（匹配MATLAB数据文件param_day_21_pv_es_ev.mat）
    if hasattr(ev_cfg, 'max_evs') and ev_cfg.max_evs is not None:
        NOFEV = min(len(EV_data), ev_cfg.max_evs)
        if NOFEV < len(EV_data):
            EV_data = EV_data[:NOFEV]  # 只使用前NOFEV个EV
            print(f"  EV数量限制: {len(EV_data)} -> {NOFEV}")
    else:
        NOFEV = len(EV_data)  # EV数量
    energy_init_ev = ev_cfg.energy_init  # MWh
    energy_end_ev = ev_cfg.energy_end  # MWh
    energy_upper_limit_ev = ev_cfg.energy_upper_limit  # MWh
    energy_lower_limit_ev = ev_cfg.energy_lower_limit  # MWh
    power_dis_upper_limit_ev = ev_cfg.power_dis_upper_limit  # MW
    power_dis_lower_limit_ev = ev_cfg.power_lower_limit
    power_ch_upper_limit_ev = ev_cfg.power_ch_upper_limit  # MW
    power_ch_lower_limit_ev = ev_cfg.power_lower_limit
    theta_ev = ev_cfg.theta
    eta_dis_ev = ev_cfg.eta_dis
    eta_ch_ev = ev_cfg.eta_ch
    pr_dis_ev = ev_cfg.pr_dis  # $/MWh
    pr_ch_ev = ev_cfg.pr_ch  # $/MWh

    # 充电状态矩阵 u (NOFEV, NOFSLOTS)
    u = np.zeros((NOFEV, NOFSLOTS))
    for idx in range(NOFEV):
        arrive_slot = int(EV_data[idx, 1]) - 1  # 转换为0索引
        depart_slot = int(EV_data[idx, 2]) - 1
        for jdx in range(NOFSLOTS):
            if arrive_slot <= jdx <= depart_slot:
                u[idx, jdx] = 1.0

    print(f"  EV原始数量: {NOFEV}")
    print(f"  在场EV数量范围: [{u.sum(axis=0).min():.0f}, {u.sum(axis=0).max():.0f}]")

    # ========== EV聚合：按(arrive, depart)分组 ==========
    ev_group_counts = None
    ev_group_mapping = None

    if hasattr(ev_cfg, 'aggregate_evs') and ev_cfg.aggregate_evs:
        u_agg, ev_group_counts, ev_group_mapping, n_groups = _aggregate_ev_by_schedule(
            EV_data, u, NOFSLOTS
        )
        print(f"  EV聚合: {NOFEV}辆 -> {n_groups}组")
        for gid, indices in ev_group_mapping.items():
            arr = int(EV_data[indices[0], 1])
            dep = int(EV_data[indices[0], 2])
            print(f"    组{gid}: {len(indices)}辆, 到达={arr}, 离开={dep}")
        u = u_agg
        NOFEV = n_groups

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

        # EV
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


if __name__ == "__main__":
    # 测试
    param, param_dict = prepare_parameters()
    print(f"\n测试通过!")
    print(f"  总资源数: {1 + 1 + param.NOFEV + param.NOFTCL + param.NOFIPP}")
