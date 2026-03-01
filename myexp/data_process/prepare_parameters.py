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

    # TCL参数
    energy_init_tcl: float
    energy_upper_limit_tcl: float
    energy_lower_limit_tcl: float
    power_ch_upper_limit_tcl: np.ndarray
    power_ch_lower_limit_tcl: np.ndarray
    theta_tcl: np.ndarray
    eta_ch_tcl: np.ndarray
    wOmiga: np.ndarray
    NOFTCL: int

    # IPP参数
    energy_init_ipp: np.ndarray
    energy_end_ipp: np.ndarray
    energy_upper_limit_ipp: np.ndarray
    energy_lower_limit_ipp: np.ndarray
    power_ch_upper_limit_ipp: np.ndarray
    power_ch_lower_limit_ipp: np.ndarray
    theta_ipp: float
    eta_ch_ipp: np.ndarray
    NOFIPP: int


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

    print(f"  EV数量: {NOFEV}")
    print(f"  在场EV数量范围: [{u.sum(axis=0).min():.0f}, {u.sum(axis=0).max():.0f}]")

    # ========== TCL参数 ==========
    tcl_cfg = config.tcl
    NOFTCL = tcl_cfg.NOFTCL  # 从配置读取TCL数量

    if NOFTCL > 0:
        tcl_c = np.array(tcl_cfg.tcl_c) * 1e-3  # 等效电容 (MWh/K)
        tcl_r = np.array(tcl_cfg.tcl_r) * 1e3  # 等效电阻 (K/MWh)
        tcl_cop = np.array(tcl_cfg.tcl_cop)  # 循环效率

        # 加载温度和热负载数据
        import scipy.io as sio
        h_load_temperature = sio.loadmat(os.path.join(base_dir, 'data_prepare', 'h_load_temperature.mat'))
        h_load_data = sio.loadmat(os.path.join(base_dir, 'data_prepare', 'h_load.mat'))

        # MATLAB: h_load = [0.4, 0.4, 0.2]' * h_load(:, 2)';
        # h_load(:, 2) 是第二列，即索引1（0-based）
        h_load_val = h_load_data['h_load'][:, 1]  # (24,) - 第二列
        h_load_temperature_val = h_load_temperature['h_load_temperature'][:, 1]  # (24,) - 第二列

        print(f"  h_load_val shape: {h_load_val.shape}")
        print(f"  h_load_temperature_val shape: {h_load_temperature_val.shape}")

        # 为每个TCL创建热负载数据
        # MATLAB: h_load = [0.4, 0.4, 0.2]' * h_load(:, 2)';
        # 结果是 (3, 24) 矩阵
        h_load = np.array(tcl_cfg.h_load_ratio).reshape(NOFTCL, 1) * h_load_val.reshape(1, -1)  # (NOFTCL, NOFSLOTS)

        # 温度转换 T' = T_ref - T
        T_ref = tcl_cfg.T_ref
        energy_init_tcl = T_ref - tcl_cfg.T_init  # 初始温度对应的能量
        energy_upper_limit_tcl = tcl_cfg.energy_upper_limit  # MWh
        energy_lower_limit_tcl = tcl_cfg.energy_lower_limit  # MWh
        power_ch_upper_limit_tcl = tcl_cfg.power_ch_upper_limit  # MW
        power_ch_lower_limit_tcl = np.zeros(NOFTCL)

        # 热力学参数
        gama = 1.0 / (tcl_c * tcl_r)
        alpha = 1.0 - gama
        beta = 1.0 / tcl_c

        theta_tcl = tcl_cfg.theta_base * alpha
        eta_ch_tcl = beta * tcl_cop

        # 热负载/外部温度影响
        # MATLAB: h_load_temperature = ones(1, NOFSLOTS) * T_ref - h_load_temperature(:, 2)';
        h_load_temperature_adj = T_ref - h_load_temperature_val  # (24,)
        # MATLAB: param.wOmiga = - repmat(beta, 1, NOFSLOTS) .* h_load - gama * h_load_temperature;
        # h_load 是 (3, 24), h_load_temperature_adj 需要广播为 (1, 24)
        wOmiga = -np.tile(beta.reshape(-1, 1), (1, NOFSLOTS)) * h_load - gama.reshape(-1, 1) * h_load_temperature_adj.reshape(1, -1)  # (NOFTCL, NOFSLOTS)

        print(f"  TCL数量: {NOFTCL}")
        print(f"  wOmiga shape: {wOmiga.shape}")
    else:
        # NOFTCL = 0 时创建默认值
        energy_init_tcl = 0.0
        energy_upper_limit_tcl = 0.0
        energy_lower_limit_tcl = 0.0
        power_ch_upper_limit_tcl = np.array([])
        power_ch_lower_limit_tcl = np.array([])
        theta_tcl = np.array([])
        eta_ch_tcl = np.array([])
        wOmiga = np.array([]).reshape(0, NOFSLOTS)
        print(f"  TCL数量: {NOFTCL}")

    # ========== IPP参数 ==========
    ipp_cfg = config.ipp
    NOFIPP = ipp_cfg.NOFIPP  # 从配置读取IPP数量

    if NOFIPP > 0:
        filename = os.path.join(base_dir, 'data_prepare', 'load_parameters_Lu_milp.xlsx')

        # 使用 pandas 读取 Excel 文件，自动处理公式
        # MATLAB: xlsread(filename) - 从第二行开始读取（跳过表头），只读取数值列
        # load_parameter(:, 1) = Excel B列 (0-based索引1) - production rate
        # load_parameter(:, 3) = Excel D列 (0-based索引3) - Pmax
        # load_parameter(:, 4) = Excel E列 (0-based索引4) - Smax
        import pandas as pd
        # skiprows=1 跳过第一行表头
        # usecols=[1, 3, 4] 读取B、D、E列
        df = pd.read_excel(filename, header=None, skiprows=1, nrows=NOFIPP, usecols=[1, 3, 4])

        # 转换为 numpy 数组，确保是浮点类型
        load_parameter = df.values.astype(float)

        # 处理可能的 NaN 值
        load_parameter = np.nan_to_num(load_parameter, nan=0.0)

        # 检查 load_parameter 的形状
        print(f"  load_parameter shape: {load_parameter.shape}")

        # 参数转换
        production_rate = 1e3 * load_parameter[:, 0]  # MW per unit material (B列, MATLAB第1列)
        S_max = load_parameter[:, 2]  # 最大材料存储容量 (E列, MATLAB第4列)
        S_tar = np.zeros(S_max.shape)
        # 瓶颈过程需要工作的工作时长
        S_tar[ipp_cfg.bottleneck_process_index] = 200 * ipp_cfg.bottleneck_working_hours

        print(f"  production_rate shape: {production_rate.shape}")
        print(f"  S_max shape: {S_max.shape}")

        energy_init_ipp = ipp_cfg.energy_init_ratio * S_max
        energy_end_ipp = energy_init_ipp + S_tar
        energy_upper_limit_ipp = S_max * ipp_cfg.energy_upper_limit_ratio
        energy_lower_limit_ipp = S_max * ipp_cfg.energy_lower_limit_ratio
        power_ch_upper_limit_ipp = 1e-3 * load_parameter[:, 1]  # MW (D列, MATLAB第3列)
        power_ch_lower_limit_ipp = ipp_cfg.power_ch_lower_limit
        theta_ipp = ipp_cfg.theta
        eta_ch_ipp = production_rate

        print(f"  IPP数量: {NOFIPP}")
    else:
        # NOFIPP = 0 时创建空数组
        energy_init_ipp = np.array([])
        energy_end_ipp = np.array([])
        energy_upper_limit_ipp = np.array([])
        energy_lower_limit_ipp = np.array([])
        power_ch_upper_limit_ipp = np.array([])
        power_ch_lower_limit_ipp = np.array([])
        theta_ipp = 1.0
        eta_ch_ipp = np.array([])
        print(f"  IPP数量: {NOFIPP}")

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

        # TCL
        energy_init_tcl=energy_init_tcl,
        energy_upper_limit_tcl=energy_upper_limit_tcl,
        energy_lower_limit_tcl=energy_lower_limit_tcl,
        power_ch_upper_limit_tcl=power_ch_upper_limit_tcl,
        power_ch_lower_limit_tcl=power_ch_lower_limit_tcl,
        theta_tcl=theta_tcl,
        eta_ch_tcl=eta_ch_tcl,
        wOmiga=wOmiga,
        NOFTCL=NOFTCL,

        # IPP
        energy_init_ipp=energy_init_ipp,
        energy_end_ipp=energy_end_ipp,
        energy_upper_limit_ipp=energy_upper_limit_ipp,
        energy_lower_limit_ipp=energy_lower_limit_ipp,
        power_ch_upper_limit_ipp=power_ch_upper_limit_ipp,
        power_ch_lower_limit_ipp=power_ch_lower_limit_ipp,
        theta_ipp=theta_ipp,
        eta_ch_ipp=eta_ch_ipp,
        NOFIPP=NOFIPP,
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
        'energy_init_tcl': energy_init_tcl,
        'energy_upper_limit_tcl': energy_upper_limit_tcl,
        'energy_lower_limit_tcl': energy_lower_limit_tcl,
        'power_ch_upper_limit_tcl': power_ch_upper_limit_tcl,
        'power_ch_lower_limit_tcl': power_ch_lower_limit_tcl,
        'theta_tcl': theta_tcl,
        'eta_ch_tcl': eta_ch_tcl,
        'wOmiga': wOmiga,
        'NOFTCL': NOFTCL,
        'energy_init_ipp': energy_init_ipp,
        'energy_end_ipp': energy_end_ipp,
        'energy_upper_limit_ipp': energy_upper_limit_ipp,
        'energy_lower_limit_ipp': energy_lower_limit_ipp,
        'power_ch_upper_limit_ipp': power_ch_upper_limit_ipp,
        'power_ch_lower_limit_ipp': power_ch_lower_limit_ipp,
        'theta_ipp': theta_ipp,
        'eta_ch_ipp': eta_ch_ipp,
        'NOFIPP': NOFIPP,
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
