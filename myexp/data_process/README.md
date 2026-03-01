# 数据准备模块 (data_process)

本模块提供完整的数据准备功能，替代原有的 MATLAB 数据处理流程。

## 模块结构

```
myexp/data_process/
├── __init__.py              # 模块入口
├── prepare_main.py           # 主入口：整合所有子模块
├── prepare_price.py          # 价格数据读取
├── prepare_regd.py          # 调频信号处理
├── prepare_pv.py            # 光伏出力数据
├── prepare_parameters.py      # 资源参数配置
├── prepare_std.py           # 参数标准化
└── example_usage.py          # 使用示例
```

## 功能说明

### 1. `prepare_price.py` - 价格数据读取
- 读取调频市场价格（容量价格、里程价格）
- 读取系统能量价格

### 2. `prepare_regd.py` - 调频信号处理
- 读取 RegD 信号数据
- 计算小时级离散分布
- 计算里程（mileage）

### 3. `prepare_pv.py` - 光伏出力数据
- 读取光伏出力预测数据

### 4. `prepare_parameters.py` - 资源参数配置
- 光伏 (PV)
- 储能 (ES)
- 电动汽车 (EV)
- 温控负载 (TCL)
- 工业生产 (IPP)

### 5. `prepare_std.py` - 参数标准化
- 将所有资源参数统一为 `(NOFDER, NOFSLOTS)` 格式
- 处理特殊约束（EV离开时段、IPP目标值等）

### 6. `prepare_main.py` - 主入口
- 整合所有子模块
- 生成完整的参数文件（兼容 .mat 格式）

## 使用方法

### 基本用法

```python
from myexp.data_process import prepare_main_data, save_parameters

# 生成参数
param, param_std, time_params, Signal_day = prepare_main_data(
    day_price=21,      # 价格日期
    hour_init=0,        # 起始小时
    NOFSLOTS=24,         # 时段数
    granularity=0.1,     # 调频信号离散粒度
    nofHisDays=14,       # 历史天数
    M=1e6,              # 大M常数
    delta_t_req=0.5,     # 维护时间
    s_perf=0.984          # 调频性能系数
)

# 保存到文件
save_parameters(
    param_market=param,
    param_std=param_std,
    time_params=time_params,
    output_path="param_day_21.mat",  # 输出路径
    day_price=21
)
```

### 完整示例

运行 `example_usage.py`:

```bash
python myexp/data_process/example_usage.py
```

## 数据格式

### 输入文件（来自 `data_prepare/` 目录）
- `07 2020.xlsx` - RegD 信号数据
- `regulation_market_results.xlsx` - 调频市场价格
- `rt_hrl_lmps.xlsx` - 系统能量价格
- `output_pv.mat` - 光伏出力数据
- `EV_arrive_leave.xlsx` - EV 到达/离开时间
- `h_load.mat` - 热负载数据
- `h_load_temperature.mat` - 外部温度数据
- `load_parameters_Lu_milp.xlsx` - IPP 参数

### 输出文件

#### `param` 市场参数
- `price_e`: (NOFSLOTS,) - 能量价格
- `price_reg`: (NOFSLOTS, 2) - 调频价格 [容量, 里程]
- `hourly_Mileage`: (NOFSLOTS,) - 小时级里程
- `hourly_Distribution`: (NOFSLOTS, NOFSCEN) - 小时级信号分布
- `d_s`: (NOFSCEN,) - 场景信号值
- `s_perf`: float - 调频性能系数

#### `param_std` 资源标准参数
所有矩阵形状为 `(NOFDER, NOFSLOTS)` 或 `(NOFDER, NOFDER)`：
- `energy_init`: (NOFDER,) - 初始能量
- `energy_upper_limit`: (NOFDER, NOFSLOTS) - 能量上限
- `energy_lower_limit`: (NOFDER, NOFSLOTS) - 能量下限
- `energy_end`: (NOFDER,) - 结束能量
- `power_dis_upper_limit`: (NOFDER, NOFSLOTS) - 放电功率上限
- `power_dis_lower_limit`: (NOFDER, NOFSLOTS) - 放电功率下限
- `power_ch_upper_limit`: (NOFDER, NOFSLOTS) - 充电功率上限
- `power_ch_lower_limit`: (NOFDER, NOFSLOTS) - 充电功率下限
- `theta`: (NOFDER,) - 保留率
- `eta_dis`: (NOFDER, NOFDER) - 放电效率矩阵
- `eta_ch`: (NOFDER, NOFDER) - 充电效率矩阵
- `pr_dis`: (NOFDER,) - 放电成本
- `pr_ch`: (NOFDER,) - 充电成本
- `wOmiga`: (NOFDER, NOFSLOTS) - 外部影响
- `index_none_reg`: (n,) - 不参与调频的资源索引

## 资源编号

```
NOFDER = 1 (PV) + 1 (ES) + NOFEV (EV) + NOFTCL (TCL) + NOFIPP (IPP)
索引范围:
- PV: [0, 0]
- ES: [1, 1]
- EV: [2, 1+NOFEV]
- TCL: [2+NOFEV, 1+NOFEV+NOFTCL]
- IPP: [2+NOFEV+NOFTCL, NOFDER-1]
```

## 与 MATLAB 版本的对比

| 功能 | MATLAB | Python |
|------|--------|--------|
| 调频信号处理 | `data_prepare_regd.m` | `prepare_regd.py` |
| 价格读取 | `data_prepare_main.m` | `prepare_price.py` |
| 光伏数据 | `data_prepare_pv_output.m` | `prepare_pv.py` |
| 资源参数 | `data_prepare_parameters.m` | `prepare_parameters.py` |
| 标准化 | `data_prepare_std.m` | `prepare_std.py` |
| 主入口 | `data_prepare_main.m` | `prepare_main.py` |

## 优势

1. **无需 MATLAB**: 纯 Python 实现，可直接运行
2. **模块化**: 每个功能独立模块，易于维护
3. **类型提示**: 使用 dataclass 和类型注解，代码更清晰
4. **可复用**: 可单独调用任何子模块
5. **兼容性**: 输出格式与 MATLAB 版本完全一致

## 依赖

- numpy >= 1.20
- scipy >= 1.5
- openpyxl >= 3.0

## 注意事项

1. 大文件读取（如 `07 2020.xlsx`）可能较慢，建议使用 pandas 优化
2. 默认使用 `data_prepare/` 目录，可通过 `base_dir` 参数修改
3. 输出的 `.mat` 文件与 MATLAB 版本兼容，可直接用于优化代码
