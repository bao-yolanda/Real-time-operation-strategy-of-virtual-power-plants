# 数据准备脚本说明

## 概述

`data_prepare.py` 是一个简化版的数据处理脚本，只处理 **PV + ES + EV**（共42个资源），移除了 TCL 和 IPP 部分。

## 功能

- 读取原始市场数据（调频价格、能量价格）
- 读取调频信号数据并计算分布
- 生成 PV、ES、EV 的物理参数
- 输出与原格式兼容的 `.mat` 文件

## 使用方法

### 方法1：直接运行测试

```bash
cd test
python test_data_prepare.py
```

这将：
1. 生成并验证数据结构
2. 保存为 `data_prepare/param_day_21.mat`

### 方法2：在代码中使用

```python
from data_prepare import data_prepare_ev_only, save_mat_file

# 生成数据
mat_data = data_prepare_ev_only(day_price=21)

# 访问数据
param = mat_data.raw['param']          # 市场参数
param_std = mat_data.raw['param_std']  # 资源参数
signal_day = mat_data.raw['Signal_day'] # 调频信号

# 保存为 .mat 文件
save_mat_file(day_price=21)
```

## 输入文件

所有输入文件位于 `data_prepare/` 目录：

| 文件名 | 说明 |
|--------|------|
| `07 2020.xlsx` | 调频信号数据（31天，每2秒一个点） |
| `output_pv.mat` | 光伏出力预测数据 |
| `EV_arrive_leave.xlsx` | 电动汽车到达/离开时间表 |
| `regulation_market_results.xlsx` | 调频市场价格（容量、里程） |
| `rt_hrl_lmps.xlsx` | 实时能量价格 |

## 输出文件

| 文件名 | 说明 |
|--------|------|
| `param_day_21.mat` | 处理后的参数文件，可直接用于 `main.py` |

## 输出数据结构

### param (市场参数)

```python
{
    'price_e': (24,),              # 实时能量价格 ($/MWh)
    'price_reg': (24, 2),         # 调频价格：第1列=容量价格，第2列=里程价格
    'hourly_Mileage': (24,),      # 每小时预期里程
    'hourly_Distribution': (24, 22), # 调频信号概率分布
    'd_s': (22,),                 # 调频信号离散级别 [-1, -0.9, ..., 1]
    'index_none_reg': array([], dtype=int), # 不参与调频的资源索引（空）
    's_perf': 0.984              # 调频性能得分
}
```

### param_std (资源参数)

```python
{
    'energy_init': (42,),          # 初始能量
    'energy_upper_limit': (42, 24), # 能量上限
    'energy_lower_limit': (42, 24), # 能量下限
    'power_dis_upper_limit': (42, 24), # 放电功率上限
    'power_dis_lower_limit': (42, 24), # 放电功率下限
    'power_ch_upper_limit': (42, 24),  # 充电功率上限
    'power_ch_lower_limit': (42, 24),  # 充电功率下限
    'theta': (42,),                # 能量保持率（全为1）
    'eta_ch': (42, 42),           # 充电效率矩阵（对角矩阵）
    'eta_dis': (42, 42),          # 放电效率矩阵（对角矩阵）
    'pr_dis': (42,),              # 放电老化成本
    'pr_ch': (42,),               # 充电老化成本
    'wOmiga': (42, 24),          # 外部影响（全为0）
}
```

### 其他参数

```python
{
    'Signal_day': (43200,),       # 调频信号序列（24小时×1800点）
    'NOFSLOTS': 24,               # 时段数
    'NOFDER': 42,                # 资源数（1 PV + 1 ES + 40 EV）
    'NOFSCEN': 22,               # 场景数
    'delta_t': 1.0,              # 时间步长（小时）
    'delta_t_req': 0.5,          # 响应时间间隔（小时）
    'M': 1e6                     # 大M常数
}
```

## 资源编号

| 资源类型 | 索引范围 | 数量 | 说明 |
|---------|---------|------|------|
| PV      | 0       | 1    | 光伏发电，只有放电功率 |
| ES      | 1       | 1    | 储能系统，充放电效率0.9 |
| EV      | 2-41    | 40   | 电动汽车，充放电效率0.9，有时段约束 |

## 与原代码的区别

| 项目 | 原代码 | 本代码 |
|-----|-------|--------|
| 资源数量 | 55 (PV+ES+40EV+3TCL+10IPP) | 42 (PV+ES+40EV) |
| theta | 部分为<1 (TCL) | 全为1 |
| wOmiga | TCL有非零值 | 全为0 |
| 数据处理 | MATLAB脚本 | Python脚本 |
| 输出格式 | .mat文件 | .mat文件（格式兼容） |

## 注意事项

1. 确保 `data_prepare/` 目录下所有输入文件都存在
2. 需要 `numpy`、`pandas`、`scipy`、`openpyxl` 等依赖
3. 生成的 `.mat` 文件可直接用于 `main.py` 中的优化算法
4. 如果想修改日期，更改 `day_price` 参数即可
