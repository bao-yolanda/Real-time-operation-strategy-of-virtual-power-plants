"""
Data preparation script for PV+ES+EV resources only.
Converts from original MATLAB data preparation to Python.
Generates .mat format data compatible with the optimization model.

Structure:
- 1 PV resource (index 1)
- 1 ES resource (index 2)
- 40 EV resources (indices 3-42)
Total: 42 resources
"""

import numpy as np
import pandas as pd
from scipy.io import savemat
import os

# Get project root directory (parent of test folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data_prepare')


def data_prepare_regd(day_price=21, data_dir=None):
    """
    Prepare RegD signal distribution from historical data.

    Args:
        day_price: Reference day for price data
        data_dir: Directory containing data files (default: DATA_DIR)

    Returns:
        param: Dictionary with RegD distribution data
    """
    if data_dir is None:
        data_dir = DATA_DIR
    NOFSLOTS = 24
    day_reg = day_price + 1  # Regulation signal day
    hour_init = 0

    # Read RegD signal data from Excel
    granularity = 0.1
    diff = granularity  # Discretization step

    filename = f'{data_dir}/07 2020.xlsx'

    # Read signal data (columns B:AF, rows 2:43202)
    signals_df = pd.read_excel(filename, sheet_name='Dynamic',
                               header=0, usecols='B:AF',
                               nrows=43201)
    signals = signals_df.values

    # Data cleaning: clip to [-1, 1]
    signals = np.clip(signals, -1, 1)

    # Process raw signal data
    nofHisDays = 14  # Use past 14 days
    signal_length = 43201 - 1  # Remove last point

    # Signal for simulation day
    Signal_day = signals[:signal_length, day_reg - 1]

    # One distribution per hour
    hourly_Distribution = []
    hourly_Mileage = []

    for hour in range(1, NOFSLOTS + 1):
        Distributions = []

        for day_idx in range(day_reg - nofHisDays, day_reg):
            # Extract column for this day
            day_signals = signals[:signal_length, day_idx - 1]

            # Initialize distribution (22 scenarios: -1 to 1)
            num_scenarios = int(2 / diff + 2)
            Distribution = np.zeros(num_scenarios)

            # Scan to get PDF
            t_start = (hour - 1) * 1800
            t_end = hour * 1800

            for t_cap in range(t_start, t_end):
                sig = day_signals[t_cap]

                if sig >= 0:  # Upward frequency adjustment
                    s_idx = int(np.ceil(sig / diff) + 1 / diff + 1)
                    if sig > 0.9999:  # Consider as 1
                        s_idx = num_scenarios - 1
                else:  # Downward
                    s_idx = int(np.floor(sig / diff) + 1 / diff + 2)
                    if sig < -0.9999:  # Consider as -1
                        s_idx = 0

                Distribution[s_idx] += 1

            # Calculate frequency
            Distribution = Distribution / np.sum(Distribution)
            Distributions.append(Distribution)

        # Average distributions across days
        Distribution = np.mean(np.array(Distributions), axis=0)
        hourly_Distribution.append(Distribution)

        # Calculate historical mileage
        Mileages = []
        for day_idx in range(day_reg - nofHisDays, day_reg):
            day_signals = signals[t_start:t_end, day_idx - 1]
            mileage = np.sum(np.abs(np.diff(day_signals)))
            Mileages.append(mileage)

        Mileage = np.mean(np.array(Mileages))
        hourly_Mileage.append(Mileage)

    # Prepare output
    param = {}
    param['hourly_Mileage'] = np.array(hourly_Mileage).reshape(-1, 1)  # 24x1
    param['hourly_Distribution'] = np.array(hourly_Distribution)  # 24x22
    param['Signal_day'] = Signal_day  # 43200 points for simulation day

    # Scenario signal values: -1, -0.9, ..., 0, ..., 0.9, 1
    d_s = np.concatenate([
        [-1],
        np.arange(-1 + 0.5 * diff, 1 - 0.5 * diff + diff, diff),
        [1]
    ]).reshape(-1, 1)
    param['d_s'] = d_s  # 22x1

    return param


def data_prepare_parameters(data_dir=None):
    """
    Prepare parameters for PV, ES, and EV resources.

    Args:
        data_dir: Directory containing data files (default: DATA_DIR)

    Returns:
        param: Dictionary with all resource parameters
    """
    if data_dir is None:
        data_dir = DATA_DIR
    NOFSLOTS = 24
    param = {}

    # ===== PV =====
    param['power_dis_upper_limit_pv'] = None  # Will be set from .mat file
    param['power_dis_lower_limit_pv'] = None

    # ===== Energy Storage =====
    param['energy_init_es'] = 1.0
    param['energy_upper_limit_es'] = 0.9 * 2.0
    param['energy_lower_limit_es'] = 0.1 * 2.0
    param['power_dis_upper_limit_es'] = 1.0
    param['power_dis_lower_limit_es'] = 0.0
    param['power_ch_upper_limit_es'] = 1.0
    param['power_ch_lower_limit_es'] = 0.0
    param['theta_es'] = 1.0
    param['eta_dis_es'] = 0.90
    param['eta_ch_es'] = 0.90
    param['pr_dis_es'] = 100.0
    param['pr_ch_es'] = 100.0

    # ===== Electric Vehicles =====
    # Read EV arrival/departure from Excel
    ev_filename = f'{data_dir}/EV_arrive_leave.xlsx'
    ev_df = pd.read_excel(ev_filename, sheet_name='EV_arrive_leave',
                          header=0, usecols='A:C', skiprows=1, nrows=120)

    EV_arrive_leave = ev_df.values  # [EV_id, arrival, departure]

    param['energy_init_ev'] = 20 * 1e-3
    param['energy_end_ev'] = 50 * 1e-3
    param['energy_upper_limit_ev'] = 60 * 0.9 * 1e-3
    param['energy_lower_limit_ev'] = 60 * 0.1 * 1e-3
    param['power_dis_upper_limit_ev'] = 22 * 1e-3
    param['power_dis_lower_limit_ev'] = 0.0
    param['power_ch_upper_limit_ev'] = 22 * 1e-3
    param['power_ch_lower_limit_ev'] = 0.0
    param['theta_ev'] = 1.0
    param['eta_dis_ev'] = 0.90
    param['eta_ch_ev'] = 0.90
    param['pr_dis_ev'] = 150.0
    param['pr_ch_ev'] = 0.0

    NOFEV = len(EV_arrive_leave)

    # Charging status u (NOFEV x 24)
    u = np.zeros((NOFEV, NOFSLOTS))
    for idx in range(NOFEV):
        arrival = int(EV_arrive_leave[idx, 1])
        departure = int(EV_arrive_leave[idx, 2])
        for jdx in range(NOFSLOTS):
            if arrival <= (jdx + 1) <= departure:
                u[idx, jdx] = 1

    param['u'] = u
    param['NOFEV'] = NOFEV
    param['EV_arrive_leave'] = EV_arrive_leave

    return param


def data_prepare_pv_output(data_dir=None):
    """
    Prepare PV output data.

    Args:
        data_dir: Directory containing data files (default: DATA_DIR)

    Returns:
        output_pv: PV output array (24,)
    """
    if data_dir is None:
        data_dir = DATA_DIR
    # Original PV data (15-minute intervals, 96 points)
    output_pv_15min = np.array([
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01,
        0.02, 0.03, 0.05, 0.10, 0.16, 0.22, 0.27, 0.32,
        0.36, 0.40, 0.44, 0.53, 0.61, 0.69, 0.77, 0.79,
        0.80, 0.80, 0.80, 0.84, 0.88, 0.93, 0.98, 1.00,
        0.99, 0.98, 0.97, 0.95, 0.91, 0.87, 0.83, 0.79,
        0.76, 0.72, 0.69, 0.65, 0.57, 0.50, 0.42, 0.35,
        0.32, 0.29, 0.26, 0.23, 0.20, 0.17, 0.13, 0.10,
        0.07, 0.05, 0.03, 0.01, 0.01, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
    ])

    # Reshape to 4x24 and accumulate to hourly
    output_pv_reshaped = output_pv_15min.reshape(4, 24)
    output_pv = output_pv_reshaped.T @ np.ones((4, 1)) * (15/60) * 2.5

    return output_pv.flatten()


def data_prepare_std(param, data_dir=None):
    """
    Standardize all resource parameters for PV+ES+EV.
    Creates unified parameter matrix: 42 x 24

    Args:
        param: Dictionary with resource parameters
        data_dir: Directory containing data files (default: DATA_DIR)

    Returns:
        param_std: Dictionary with standardized parameters
    """
    if data_dir is None:
        data_dir = DATA_DIR
    NOFEV = param['NOFEV']
    NOFSLOTS = 24
    NOFDER = 1 + 1 + NOFEV  # PV + ES + EV = 42

    # Load PV output
    output_pv = data_prepare_pv_output(data_dir)

    param_std = {}

    # ===== Initial Energy (42 x 1) =====
    energy_init = np.zeros((NOFDER, 1))
    energy_init[0] = 0  # PV: no energy
    energy_init[1] = param['energy_init_es']
    energy_init[2:] = param['energy_init_ev']
    param_std['energy_init'] = energy_init

    # ===== Energy Upper Limit (42 x 24) =====
    energy_upper_limit = np.zeros((NOFDER, 1))
    energy_upper_limit[0] = 2  # PV: max output
    energy_upper_limit[1] = param['energy_upper_limit_es']
    energy_upper_limit[2:] = param['energy_upper_limit_ev']
    param_std['energy_upper_limit'] = np.tile(energy_upper_limit, (1, NOFSLOTS))

    # ===== Energy End (42 x 1) =====
    energy_end = np.zeros((NOFDER, 1))
    energy_end[0] = 0
    energy_end[1] = param['energy_init_es']
    energy_end[2:] = param['energy_end_ev']
    param_std['energy_end'] = energy_end

    # ===== Energy Lower Limit (42 x 24) =====
    energy_lower_limit = np.zeros((NOFDER, 1))
    energy_lower_limit[0] = 0
    energy_lower_limit[1] = param['energy_lower_limit_es']
    energy_lower_limit[2:] = param['energy_lower_limit_ev']
    param_std['energy_lower_limit'] = np.concatenate([
        np.tile(energy_lower_limit, (1, NOFSLOTS - 1)),
        energy_end
    ], axis=1)

    # ===== Discharge Power Upper Limit (42 x 24) =====
    power_dis_upper_limit = np.zeros((NOFDER, NOFSLOTS))
    power_dis_upper_limit[0, :] = output_pv  # PV
    power_dis_upper_limit[1, :] = param['power_dis_upper_limit_es']  # ES
    power_dis_upper_limit[2:, :] = (param['power_dis_upper_limit_ev'] *
                                     param['u'])  # EV
    param_std['power_dis_upper_limit'] = power_dis_upper_limit

    # ===== Discharge Power Lower Limit (42 x 24) =====
    param_std['power_dis_lower_limit'] = np.zeros((NOFDER, NOFSLOTS))

    # ===== Charge Power Upper Limit (42 x 24) =====
    power_ch_upper_limit = np.zeros((NOFDER, NOFSLOTS))
    power_ch_upper_limit[0, :] = 0  # PV: no charge
    power_ch_upper_limit[1, :] = param['power_ch_upper_limit_es']  # ES
    power_ch_upper_limit[2:, :] = (param['power_ch_upper_limit_ev'] *
                                    param['u'])  # EV
    param_std['power_ch_upper_limit'] = power_ch_upper_limit

    # ===== Charge Power Lower Limit (42 x 24) =====
    param_std['power_ch_lower_limit'] = np.zeros((NOFDER, NOFSLOTS))

    # ===== Retention Rate (42 x 1) =====
    theta = np.ones((NOFDER, 1))  # All have theta=1
    param_std['theta'] = theta

    # ===== Discharge Efficiency (42 x 42 diagonal matrix) =====
    eta_dis = np.zeros((NOFDER, NOFDER))
    eta_dis[1, 1] = 1 / param['eta_dis_es']
    for idx in range(2, 2 + NOFEV):
        eta_dis[idx, idx] = 1 / param['eta_dis_ev']
    param_std['eta_dis'] = eta_dis

    # ===== Charge Efficiency (42 x 42 diagonal matrix) =====
    eta_ch = np.zeros((NOFDER, NOFDER))
    eta_ch[1, 1] = param['eta_ch_es']
    for idx in range(2, 2 + NOFEV):
        eta_ch[idx, idx] = param['eta_ch_ev']
    param_std['eta_ch'] = eta_ch

    # ===== Discharge Cost (42 x 1) =====
    pr_dis = np.zeros((NOFDER, 1))
    pr_dis[1] = param['pr_dis_es']
    pr_dis[2:] = param['pr_dis_ev']
    param_std['pr_dis'] = pr_dis

    # ===== Charge Cost (42 x 1) =====
    pr_ch = np.zeros((NOFDER, 1))
    pr_ch[1] = param['pr_ch_es']
    pr_ch[2:] = param['pr_ch_ev']
    param_std['pr_ch'] = pr_ch

    # ===== External Influences (42 x 24) =====
    # PV, ES, EV have no external influences
    param_std['wOmiga'] = np.zeros((NOFDER, NOFSLOTS))

    # ===== Modify EV minimum energy at departure =====
    for idx in range(NOFEV):
        for jdx in range(NOFSLOTS - 1):
            if param['u'][idx, jdx] - param['u'][idx, jdx + 1] == 1:
                # Last time slot
                param_std['energy_lower_limit'][2 + idx, jdx] = \
                    param_std['energy_end'][2 + idx, 0]

    # Modify EV minimum energy before departure
    for idx in range(NOFEV):
        for jdx in range(1, NOFSLOTS - 1):
            if param['u'][idx, jdx] - param['u'][idx, jdx + 1] == 1:
                # One slot before last
                param_std['energy_lower_limit'][2 + idx, jdx - 1] = \
                    param_std['energy_end'][2 + idx, 0] - \
                    param_std['power_ch_upper_limit'][2 + idx, jdx] * 0.9

                # Two slots before last
                if jdx >= 2:
                    param_std['energy_lower_limit'][2 + idx, jdx - 2] = \
                        param_std['energy_end'][2 + idx, 0] - \
                        param_std['power_ch_upper_limit'][2 + idx, jdx] * 0.9 * 2

    return param_std


def read_market_prices(day_price=21, data_dir=None):
    """
    Read regulation market and energy prices.

    Args:
        day_price: Reference day
        data_dir: Directory containing data files (default: DATA_DIR)

    Returns:
        price_reg: Regulation prices [capacity_price, mileage_price] (24 x 2)
        price_e: Energy prices (24 x 1)
    """
    if data_dir is None:
        data_dir = DATA_DIR
    NOFSLOTS = 24
    hour_init = 0

    # Read regulation market prices
    reg_filename = f'{data_dir}/regulation_market_results.xlsx'
    reg_df = pd.read_excel(reg_filename, sheet_name='regulation_market_results',
                           header=0, usecols='G:H')

    start_row = (day_price - 1) * 24 + hour_init + 1  # 0-indexed
    price_reg = reg_df.iloc[start_row:start_row + NOFSLOTS].values.astype(float)

    # Read energy prices
    energy_filename = f'{data_dir}/rt_hrl_lmps.xlsx'
    energy_df = pd.read_excel(energy_filename, sheet_name='rt_hrl_lmps',
                              header=0, usecols='I')

    price_e = energy_df.iloc[start_row:start_row + NOFSLOTS].values.astype(float).reshape(-1, 1)

    return price_reg, price_e


def data_prepare_main(day_price=21, output_file='param_day_21_pv_es_ev.mat',
                      data_dir=None, output_dir='./'):
    """
    Main data preparation function for PV+ES+EV.

    Args:
        day_price: Day index for price data
        output_file: Output .mat filename
        data_dir: Directory containing input data files (default: DATA_DIR)
        output_dir: Directory for output .mat file
    """
    if data_dir is None:
        data_dir = DATA_DIR
    print(f"Preparing data for day {day_price}...")

    # Prepare RegD distribution
    print("Processing RegD signal distribution...")
    param_reg = data_prepare_regd(day_price, data_dir)

    # Prepare resource parameters
    print("Processing resource parameters...")
    param_res = data_prepare_parameters(data_dir)
    nofev = param_res['NOFEV']

    # Read market prices
    print("Reading market prices...")
    price_reg, price_e = read_market_prices(day_price, data_dir)

    # Standardize parameters
    print("Standardizing parameters...")
    param_std = data_prepare_std(param_res, data_dir)

    # Compile final parameter dictionary
    print("Compiling final parameters...")

    # Param (regulation and market data, plus basic resource parameters)
    param = {
        # Regulation signal distribution
        'hourly_Distribution': param_reg['hourly_Distribution'],
        'hourly_Mileage': param_reg['hourly_Mileage'],
        'd_s': param_reg['d_s'],

        # Market prices
        'price_reg': price_reg,
        'price_e': price_e,
        's_perf': 0.984,

        # Resource names and range
        'resource_names': ['pv', 'es', 'ev'],
        'resource_range': np.array([[1, 1], [2, 2], [3, nofev + 2]]),

        # Index of non-regulation resources (empty for PV+ES+EV only)
        'index_none_reg': np.array([]),
    }

    # Param_std (standardized resource parameters)
    param_std_dict = {
        'energy_init': param_std['energy_init'],
        'energy_upper_limit': param_std['energy_upper_limit'],
        'energy_end': param_std['energy_end'],
        'energy_lower_limit': param_std['energy_lower_limit'],
        'power_dis_upper_limit': param_std['power_dis_upper_limit'],
        'power_dis_lower_limit': param_std['power_dis_lower_limit'],
        'power_ch_upper_limit': param_std['power_ch_upper_limit'],
        'power_ch_lower_limit': param_std['power_ch_lower_limit'],
        'theta': param_std['theta'],
        'eta_dis': param_std['eta_dis'],
        'eta_ch': param_std['eta_ch'],
        'pr_dis': param_std['pr_dis'],
        'pr_ch': param_std['pr_ch'],
        'wOmiga': param_std['wOmiga'],
    }

    # Final data dictionary (matching original MATLAB .mat structure)
    param_final = {
        'param': param,
        'param_std': param_std_dict,

        # Constants
        'M': 1e6,
        'delta_t_req': 0.5,
        'delta_t': 1.0,
        'NOFSLOTS': 24,
        'NOFDER': 1 + 1 + nofev,  # PV + ES + EV
        'NOFSCEN': len(param_reg['hourly_Distribution'][0, :]),

        # Signal for simulation day
        'Signal_day': param_reg['Signal_day'],

        # Resource count
        'NOFEV': nofev,
    }

    # Save to .mat file
    output_path = f'{output_dir}/{output_file}'
    print(f"Saving to {output_path}...")
    savemat(output_path, param_final)

    print("Data preparation completed successfully!")
    print(f"  Resources: {param_final['NOFDER']} (1 PV + 1 ES + {param_final['NOFEV']} EV)")
    print(f"  Time slots: {param_final['NOFSLOTS']}")
    print(f"  Scenarios: {param_final['NOFSCEN']}")

    return param_final


if __name__ == '__main__':
    # Generate data for day 21
    param = data_prepare_main(
        day_price=21,
        output_file='test.mat',
        data_dir=DATA_DIR,
        output_dir='./'
    )

