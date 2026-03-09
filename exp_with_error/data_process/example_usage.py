"""
示例：使用 data_process 模块生成数据
"""
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myexp.data_process import prepare_main_data


def main():
    """主函数：生成参数数据"""

    # 配置参数
    day_price = 21  # 价格日期
    hour_init = 0  # 起始小时
    NOFSLOTS = 24  # 时段数（24小时）
    granularity = 0.1  # 调频信号离散粒度
    nofHisDays = 14  # 历史天数
    M = 1e6  # 大M常数
    delta_t_req = 0.5  # 维护时间
    s_perf = 0.984  # 调频性能系数

    # 生成数据
    param, param_std, time_params, Signal_day = prepare_main_data(
        day_price=day_price,
        hour_init=hour_init,
        NOFSLOTS=NOFSLOTS,
        granularity=granularity,
        nofHisDays=nofHisDays,
        M=M,
        delta_t_req=delta_t_req,
        s_perf=s_perf
    )

    # 打印关键信息
    print("\n" + "="*70)
    print("数据生成完成！")
    print("="*70)
    print("\n关键参数:")
    print(f"  NOFSLOTS (时段数): {time_params['NOFSLOTS']}")
    print(f"  NOFSCEN (场景数): {time_params['NOFSCEN']}")
    print(f"  NOFDER (资源数): {time_params['NOFDER']}")

    print("\n价格范围:")
    print(f"  能量价格: [{param.price_e.min():.2f}, {param.price_e.max():.2f}] $/MWh")
    print(f"  容量价格: [{param.price_reg[:, 0].min():.2f}, {param.price_reg[:, 0].max():.2f}] $/MW")
    print(f"  里程价格: [{param.price_reg[:, 1].min():.2f}, {param.price_reg[:, 1].max():.2f}] $/MW")

    print("\n调频信号:")
    print(f"  d_s 范围: [{param.d_s.min():.4f}, {param.d_s.max():.4f}]")
    print(f"  里程范围: [{param.hourly_Mileage.min():.4f}, {param.hourly_Mileage.max():.4f}]")

    print("\n能量约束验证:")
    print(f"  energy_init shape: {param_std.energy_init.shape}")
    print(f"  energy_upper_limit shape: {param_std.energy_upper_limit.shape}")
    print(f"  power_dis_upper_limit shape: {param_std.power_dis_upper_limit.shape}")


if __name__ == "__main__":
    main()
