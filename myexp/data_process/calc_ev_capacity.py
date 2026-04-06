"""计算EV总充电容量"""
from myexp.data_process.prepare_parameters import prepare_parameters, EV_TYPES
from collections import Counter

# 获取默认参数 (target_ev_count=120)
param, _ = prepare_parameters(target_ev_count=120)

# 统计各类型EV数量
type_counts = Counter(param.ev_types)
print('=' * 60)
print('EV总充电容量计算')
print('=' * 60)

# 各类型参数
print('\n各类型EV参数:')
total_charge_need = 0
total_battery_capacity = 0

for t in ['a', 'b', 'c']:
    cfg = EV_TYPES[t]
    count = type_counts.get(t, 0)
    
    # 单辆EV充电需求 (kWh)
    charge_need_kwh = cfg.battery_capacity * (cfg.energy_end_ratio - cfg.energy_init_ratio)
    charge_need_mwh = charge_need_kwh * 1e-3
    
    # 该类型总充电需求
    type_total_kwh = count * charge_need_kwh
    type_total_mwh = count * charge_need_mwh
    
    # 该类型总电池容量
    type_battery_kwh = count * cfg.battery_capacity
    
    total_charge_need += type_total_kwh
    total_battery_capacity += type_battery_kwh
    
    print(f'\n类型 {t}: {count} 辆')
    print(f'  电池容量: {cfg.battery_capacity} kWh')
    print(f'  SOC: {cfg.energy_init_ratio*100:.1f}% -> {cfg.energy_end_ratio*100:.1f}%')
    print(f'  单辆充电需求: {charge_need_kwh:.1f} kWh')
    print(f'  单辆最大充电功率: {cfg.power_ch_limit} kW')
    print(f'  类型总充电需求: {type_total_kwh:.1f} kWh = {type_total_mwh:.3f} MWh')
    print(f'  类型总电池容量: {type_battery_kwh:.1f} kWh')

print('\n' + '=' * 60)
print('汇总:')
print(f'  EV总数: {param.NOFEV} 辆')
print(f'  总电池容量: {total_battery_capacity:.1f} kWh = {total_battery_capacity/1000:.2f} MWh')
print(f'  总充电需求: {total_charge_need:.1f} kWh = {total_charge_need/1000:.2f} MWh')
print('=' * 60)
