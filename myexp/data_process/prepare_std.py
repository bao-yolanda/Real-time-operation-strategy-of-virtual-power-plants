"""
参数标准化
将所有资源参数统一格式为 NOFDER x NOFSLOTS 的矩阵
支持多类型EV的向量化参数
"""
import numpy as np
from typing import Dict
from .prepare_parameters import ResourceParameters
from .config import ResourceConfig, default_config


def prepare_std_parameters(
    param: ResourceParameters,
    NOFSLOTS: int = None,
    config: ResourceConfig = None
) -> Dict[str, np.ndarray]:
    """
    标准化资源参数

    Args:
        param: 资源参数对象
        NOFSLOTS: 时段数量,如果为None则使用配置文件中的默认值
        config: 资源配置对象,如果为None则使用默认配置

    Returns:
        param_std: 标准化参数字典
            - 所有矩阵的形状为 (NOFDER, NOFSLOTS)
            - NOFDER = 1(PV) + 1(ES) + NOFEV(EV)
    """
    if config is None:
        config = default_config

    if NOFSLOTS is None:
        NOFSLOTS = config.system.NOFSLOTS

    NOFPV = config.pv.NOFPV 
    NOFEV = param.NOFEV
    NOFDER = NOFPV + 1 + NOFEV

    # EV聚合系数：如果有聚合，每组乘以组内车辆数；否则每辆EV系数=1
    ev_counts = param.ev_group_counts if param.ev_group_counts is not None else np.ones(NOFEV, dtype=int)

    print(f"  标准化参数: NOFDER={NOFDER}, NOFSLOTS={NOFSLOTS}"
          f"{f', EV聚合({NOFEV}组)' if param.ev_group_counts is not None else ''}")

    # ========== 初始能量 ==========
    # 注意：EV参数现在是数组 (NOFEV,)
    pv_cfg = config.pv
    energy_init_list = [
        np.array([pv_cfg.energy_init]),  # PV初始能量
        np.array([param.energy_init_es]),  # ES
        param.energy_init_ev,  # EV (NOFEV,) 已经是数组
    ]
    energy_init = np.concatenate(energy_init_list)
    param_std = {'energy_init': energy_init}

    # ========== 能量上限 ==========
    # param.energy_upper_limit_ev 是 (NOFEV,) 数组
    # 需要扩展为 (NOFEV, NOFSLOTS) 矩阵
    energy_upper_limit_list = [
        np.full(NOFSLOTS, pv_cfg.energy_upper_limit),  # PV能量上限
        np.full(NOFSLOTS, param.energy_upper_limit_es),
        np.tile(param.energy_upper_limit_ev.reshape(-1, 1), (1, NOFSLOTS)),  # EV (NOFEV, NOFSLOTS)
    ]
    energy_upper_limit = np.vstack(energy_upper_limit_list)
    param_std['energy_upper_limit'] = energy_upper_limit

    # ========== 结束能量（也是下限的一部分） ==========
    # 使用 concatenate 而不是 array，因为元素形状不同
    energy_end_list = [
        np.array([pv_cfg.energy_end]),  # PV 标量转数组
        np.array([param.energy_init_es]),  # ES回到初始值
        param.energy_end_ev,  # EV (NOFEV,) 数组
    ]
    energy_end = np.concatenate(energy_end_list)
    param_std['energy_end'] = energy_end

    # ========== 能量下限 ==========
    energy_lower_limit_base_list = [
        np.full(NOFSLOTS, pv_cfg.energy_lower_limit),  # PV
        np.full(NOFSLOTS, param.energy_lower_limit_es),
        np.tile(param.energy_lower_limit_ev.reshape(-1, 1), (1, NOFSLOTS)),  # EV (NOFEV, NOFSLOTS)
    ]
    energy_lower_limit_base = np.vstack(energy_lower_limit_base_list)

    # 最后一个时段的下限 = energy_end
    energy_lower_limit = np.hstack([
        energy_lower_limit_base[:, :NOFSLOTS-1],
        energy_end.reshape(-1, 1)
    ])
    param_std['energy_lower_limit'] = energy_lower_limit

    # ========== 放电功率上限 ==========
    # param.power_dis_upper_limit_ev 是 (NOFEV,) 数组
    # param.u 是 (NOFEV, NOFSLOTS) 矩阵
    power_dis_upper_limit = np.vstack([
        param.power_dis_upper_limit_pv.reshape(1, -1),  # PV (NOFPV, NOFSLOTS)
        np.full((1, NOFSLOTS), param.power_dis_upper_limit_es),  # ES (1, NOFSLOTS)
        np.tile(param.power_dis_upper_limit_ev.reshape(-1, 1), (1, NOFSLOTS)) * param.u,  # EV
    ])
    param_std['power_dis_upper_limit'] = power_dis_upper_limit

    # ========== 放电功率下限 ==========
    power_dis_lower_limit = np.zeros_like(power_dis_upper_limit)
    param_std['power_dis_lower_limit'] = power_dis_lower_limit

    # ========== 充电功率上限 ==========
    power_ch_upper_limit_list = [
        np.zeros((NOFPV, NOFSLOTS)),  # PV不能充电
        np.full((1, NOFSLOTS), param.power_ch_upper_limit_es),  # ES
        np.tile(param.power_ch_upper_limit_ev.reshape(-1, 1), (1, NOFSLOTS)) * param.u,  # EV
    ]
    power_ch_upper_limit = np.vstack(power_ch_upper_limit_list)
    param_std['power_ch_upper_limit'] = power_ch_upper_limit

    # ========== 充电功率下限 ==========
    power_ch_lower_limit = np.zeros_like(power_ch_upper_limit)
    param_std['power_ch_lower_limit'] = power_ch_lower_limit

    # ========== 保留率 theta ==========
    theta_list = [
        np.full(NOFPV, 1),  # PV
        np.array([param.theta_es]),  # ES
        np.full(NOFEV, param.theta_ev),  # EV
    ]
    theta = np.concatenate(theta_list)
    param_std['theta'] = theta

    # ========== 放电效率 eta_dis ==========
    eta_dis = np.zeros((NOFDER, NOFDER))
    eta_dis[NOFPV, NOFPV] = 1 / param.eta_dis_es  # ES
    # EV: param.eta_dis_ev 是 (NOFEV,) 数组
    for idx in range(NOFPV + 1, NOFPV + 1 + NOFEV):
        ev_idx = idx - NOFPV - 1  # EV在数组中的索引
        eta_dis[idx, idx] = 1 / param.eta_dis_ev[ev_idx]
    param_std['eta_dis'] = eta_dis

    # ========== 充电效率 eta_ch ==========
    eta_ch = np.zeros((NOFDER, NOFDER))
    eta_ch[NOFPV, NOFPV] = param.eta_ch_es  # ES
    for idx in range(NOFPV + 1, NOFPV + 1 + NOFEV):
        ev_idx = idx - NOFPV - 1
        eta_ch[idx, idx] = param.eta_ch_ev[ev_idx]
    param_std['eta_ch'] = eta_ch

    # ========== 放电成本 ==========
    # param.pr_dis_ev 是 (NOFEV,) 数组
    pr_dis_list = [
        np.zeros(NOFPV),  # PV没有放电成本
        np.array([param.pr_dis_es]),
        param.pr_dis_ev,  # EV (NOFEV,) 数组
    ]
    pr_dis = np.concatenate(pr_dis_list)
    param_std['pr_dis'] = pr_dis

    # ========== 充电成本 ==========
    pr_ch_list = [
        np.zeros(NOFPV),  # PV
        np.array([param.pr_ch_es]),
        param.pr_ch_ev,  # EV (NOFEV,) 数组
    ]
    pr_ch = np.concatenate(pr_ch_list)
    param_std['pr_ch'] = pr_ch

    # ========== 外部影响 wOmiga ==========
    wOmiga_list = [
        np.zeros((NOFPV, NOFSLOTS)),  # PV没有外部影响
        np.zeros((1, NOFSLOTS)),  # ES没有外部影响
        np.zeros((NOFEV, NOFSLOTS)),  # EV没有外部影响
    ]
    wOmiga = np.vstack(wOmiga_list)
    param_std['wOmiga'] = wOmiga

    # ========== EV特殊处理：离开时段前后的能量下限 ==========
    ev_cfg = config.ev
    
    # 第一个循环：最后一个可充电时段的能量下限 = energy_end
    for idx in range(NOFEV):
        ev_idx = NOFPV + 1 + idx  # EV在param_std中的索引
        for jdx in range(NOFSLOTS - 1):
            # jdx是0-based, 从0到22 (NOFSLOTS-2)
            # u的变化从1到0表示离开
            if param.u[idx, jdx] - param.u[idx, jdx + 1] == 1:  # jdx+1是离开时段
                # jdx是最后一个在场时段
                param_std['energy_lower_limit'][ev_idx, jdx] = param_std['energy_end'][ev_idx]

    # 第二个循环：离开前两个时段的能量下限需要考虑充电能力
    for idx in range(NOFEV):
        ev_idx = NOFPV + 1 + idx  # EV在param_std中的索引
        for jdx in range(NOFSLOTS - 1):
            # 找到最后一个在场时段
            if param.u[idx, jdx] - param.u[idx, jdx + 1] == 1:
                # jdx是最后一个在场时段，jdx+1是离开时段
                # 设置jdx-1和jdx-2的能量下限
                pre_departure_factor = ev_cfg.pre_departure_energy_factor
                if jdx - 1 >= 0:
                    param_std['energy_lower_limit'][ev_idx, jdx - 1] = (
                        param_std['energy_end'][ev_idx] -
                        param_std['power_ch_upper_limit'][ev_idx, jdx] * pre_departure_factor
                    )
                if jdx - 2 >= 0:
                    param_std['energy_lower_limit'][ev_idx, jdx - 2] = (
                        param_std['energy_end'][ev_idx] -
                        param_std['power_ch_upper_limit'][ev_idx, jdx] * pre_departure_factor * 2
                    )

    # 验证维度
    print(f"  energy_init shape: {param_std['energy_init'].shape}")
    print(f"  energy_upper_limit shape: {param_std['energy_upper_limit'].shape}")
    print(f"  power_dis_upper_limit shape: {param_std['power_dis_upper_limit'].shape}")
    print(f"  power_ch_upper_limit shape: {param_std['power_ch_upper_limit'].shape}")
    print(f"  eta_ch shape: {param_std['eta_ch'].shape}")

    return param_std


if __name__ == "__main__":
    from .prepare_parameters import prepare_parameters
    param, _ = prepare_parameters()
    param_std = prepare_std_parameters(param)
    print(f"\n标准化完成!")
