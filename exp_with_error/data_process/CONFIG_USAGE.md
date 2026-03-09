# 配置文件使用说明

## 概述

为了避免代码中的硬编码,我们引入了配置文件系统。所有关键的系统参数都可以通过配置文件进行管理,提高了代码的可维护性和可配置性。

## 配置文件结构

配置文件位于 `myexp/data_process/config.py`,包含以下配置类:

### 1. PVConfig - 光伏配置
```python
@dataclass
class PVConfig:
    NOFPV: int = 1              # PV数量
    energy_init: float = 1.0    # PV初始能量(MWh)
    energy_end: float = 0.0     # PV结束能量(MWh)
    energy_upper_limit: float = 2.0   # PV能量上限(MWh)
    energy_lower_limit: float = 0.0   # PV能量下限(MWh)
```

### 2. ESConfig - 储能系统配置
```python
@dataclass
class ESConfig:
    energy_init: float = 1.0                # 初始能量(MWh)
    energy_upper_limit_ratio: float = 0.9   # 上限比例
    energy_capacity: float = 2.0            # 总容量(MWh)
    energy_lower_limit_ratio: float = 0.1   # 下限比例
    power_dis_upper_limit: float = 1.0     # 放电功率上限(MW)
    power_dis_lower_limit: float = 0.0     # 放电功率下限(MW)
    power_ch_upper_limit: float = 1.0      # 充电功率上限(MW)
    power_ch_lower_limit: float = 0.0      # 充电功率下限(MW)
    theta: float = 1.0                     # 能量保留率
    eta_dis: float = 0.90                  # 放电效率
    eta_ch: float = 0.90                   # 充电效率
    pr_dis: float = 100.0                  # 放电成本($/MWh)
    pr_ch: float = 100.0                   # 充电成本($/MWh)
```

### 3. EVConfig - 电动汽车配置
```python
@dataclass
class EVConfig:
    battery_capacity: float = 60.0         # 电池容量(kWh)
    energy_init_ratio: float = 20/60       # 初始SOC比例
    energy_end_ratio: float = 50/60        # 结束SOC比例
    energy_upper_limit_ratio: float = 0.9  # 上限比例
    energy_lower_limit_ratio: float = 0.1  # 下限比例
    power_dis_limit: float = 22.0          # 放电功率限制(kW)
    power_ch_limit: float = 22.0           # 充电功率限制(kW)
    power_lower_limit: float = 0.0         # 功率下限(kW)
    theta: float = 1.0                     # 能量保留率
    eta_dis: float = 0.90                  # 放电效率
    eta_ch: float = 0.90                   # 充电效率
    pr_dis: float = 150.0                  # 放电成本($/MWh)
    pr_ch: float = 0.0                     # 充电成本($/MWh)
    pre_departure_energy_factor: float = 0.9  # 离开前能量调整系数
```

### 4. TCLConfig - 温控负载配置
```python
@dataclass
class TCLConfig:
    NOFTCL: int = 3                        # TCL数量
    tcl_c: List[float] = [80, 80, 40]     # 等效电容(MWh/K)
    tcl_r: List[float] = [0.1, 0.1, 0.15] # 等效电阻(K/MWh)
    tcl_cop: List[float] = [3.6, 3.6, 3.3]# 循环效率
    h_load_ratio: List[float] = [0.4, 0.4, 0.2]  # 热负载分配比例
    T_ref: float = 28.0                   # 参考温度(K)
    T_init: float = 26.0                   # 初始温度(K)
    energy_upper_limit: float = 4.0        # 能量上限(MWh)
    energy_lower_limit: float = 0.0        # 能量下限(MWh)
    power_ch_limit: List[float] = [400, 400, 200]  # 充电功率限制(kW)
```

### 5. IPPConfig - 工业负荷配置
```python
@dataclass
class IPPConfig:
    NOFIPP: int = 10                       # IPP数量
    energy_init_ratio: float = 0.5         # 初始存储比例
    energy_upper_limit_ratio: float = 0.90  # 上限比例
    energy_lower_limit_ratio: float = 0.10  # 下限比例
    power_lower_limit: float = 0.0         # 功率下限(MW)
    theta: float = 1.0                      # 能量保留率
    final_slots_count: int = 14            # 最后时段计算数量
    bottleneck_working_hours: float = 22.0  # 瓶颈过程工作时长
    bottleneck_process_index: int = 0      # 瓶颈过程索引
```

### 6. SystemConfig - 系统配置
```python
@dataclass
class SystemConfig:
    NOFSLOTS: int = 24                     # 时段数量
    none_reg_offsets: List[int] = [1,2,3,6,7]  # 不参与调频资源的索引偏移
```

## 使用方法

### 方法1: 使用默认配置

```python
from myexp.data_process import prepare_main_data

# 使用默认配置
param, param_std, time_params, Signal_day = prepare_main_data(day_price=21)
```

### 方法2: 使用自定义配置

```python
from myexp.data_process import prepare_main_data
from myexp.data_process.config import ResourceConfig, PVConfig, ESConfig, EVConfig

# 创建自定义配置
custom_config = ResourceConfig(
    pv=PVConfig(NOFPV=2),  # 使用2个PV
    ev=EVConfig(battery_capacity=80.0),  # EV电池容量80kWh
    es=ESConfig(energy_capacity=3.0),    # ES容量3MWh
    system=SystemConfig(NOFSLOTS=48)     # 48个时段
)

# 使用自定义配置
param, param_std, time_params, Signal_day = prepare_main_data(
    day_price=21,
    config=custom_config
)
```

### 方法3: 修改配置文件

直接编辑 `myexp/data_process/config.py` 中的默认值:

```python
# 在 config.py 中修改
@dataclass
class PVConfig:
    NOFPV: int = 5  # 修改为5个PV
    energy_upper_limit: float = 3.0  # 修改能量上限
```

## 配置参数说明

### 避免硬编码的好处

1. **集中管理**: 所有参数在一个地方定义,方便查找和修改
2. **类型安全**: 使用dataclass类型注解,IDE可以提供智能提示
3. **易于测试**: 可以轻松创建不同的测试配置
4. **文档化**: 配置类本身就是最好的文档
5. **可重用**: 配置可以在多个模块之间共享

### 参数计算属性

某些参数使用`@property`装饰器自动计算:

```python
# ESConfig 示例
@property
def energy_upper_limit(self) -> float:
    return self.energy_upper_limit_ratio * self.energy_capacity

# 使用时直接访问
es_config = ESConfig()
upper_limit = es_config.energy_upper_limit  # 自动计算 0.9 * 2.0 = 1.8
```

### EV参数示例

```python
ev_config = EVConfig(
    battery_capacity=60.0,  # kWh
    energy_init_ratio=20/60,  # 初始SOC
    energy_end_ratio=50/60   # 结束SOC
)

# 自动计算的能量值(MWh)
ev_config.energy_init   # 60 * (20/60) * 1e-3 = 0.02
ev_config.energy_end     # 60 * (50/60) * 1e-3 = 0.05
ev_config.energy_upper_limit  # 60 * 0.9 * 1e-3 = 0.054
```

## 常见配置场景

### 场景1: 无TCL和IPP

```python
from myexp.data_process.config import ResourceConfig, TCLConfig, IPPConfig

custom_config = ResourceConfig(
    tcl=TCLConfig(NOFTCL=0),  # 无TCL
    ipp=IPPConfig(NOFIPP=0)   # 无IPP
)
```

### 场景2: 更多的EV

```python
# 修改EV配置文件中的参数,或者在运行时创建自定义配置
from myexp.data_process.config import ResourceConfig, EVConfig

custom_config = ResourceConfig(
    ev=EVConfig(
        battery_capacity=80.0,
        power_ch_limit=40.0,
        power_dis_limit=40.0
    )
)
```

### 场景3: 不同的时间分辨率

```python
from myexp.data_process.config import ResourceConfig, SystemConfig

custom_config = ResourceConfig(
    system=SystemConfig(NOFSLOTS=48)  # 半小时间段
)
```

## 注意事项

1. **配置一致性**: 修改配置时,确保相关参数之间的逻辑一致性
   - 例如: `energy_upper_limit_ratio` 和 `energy_lower_limit_ratio` 之和不应超过1

2. **单位转换**: 配置中使用了不同的单位(MW, kW, MWh, kWh),内部会自动转换
   - 功率: MW (配置可使用kW,会自动 * 1e-3 转换)
   - 能量: MWh (配置可使用kWh,会自动 * 1e-3 转换)

3. **索引系统**: Python使用0-based索引,Matlab使用1-based索引
   - 配置文件中的所有索引都是0-based
   - 代码中已经处理了与Matlab的索引转换

4. **默认配置**: `default_config` 是全局单例,谨慎直接修改
   - 如果需要临时修改,创建新的 `ResourceConfig` 实例
   - 如果需要永久修改,编辑 `config.py` 中的默认值

## 总结

通过配置文件系统,我们实现了:

✅ 消除硬编码,提高代码可维护性
✅ 集中管理所有系统参数
✅ 支持灵活的参数配置
✅ 保持与Matlab实现的一致性
✅ 提供类型安全和IDE智能提示
