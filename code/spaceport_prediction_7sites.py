# -*- coding: utf-8 -*-
"""
火箭发射基地发射量预测 (2050年及之后) - 优化版
移除低效发射站: Baikonur, Kourou, Satish Dhawan
保留7个主要活跃发射基地

使用GM(1,1)灰色预测模型 + 逻辑增长约束
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 历史数据 (2015-2024年) - 保留7个活跃发射基地
# 移除: Baikonur(高纬度45.6°N, 低发射量), Kourou(发射量骤降), Satish Dhawan(无增长)
# ============================================================================
years = np.array([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])

spaceport_data = {
    'Cape Canaveral': np.array([17, 18, 7, 17, 13, 20, 19, 38, 59, 67]),   # 28.5°N, 主力
    'Kennedy': np.array([0, 0, 12, 3, 3, 10, 12, 19, 13, 26]),              # 28.5°N, NASA主基地
    'Vandenberg': np.array([7, 3, 9, 9, 7, 5, 10, 16, 30, 47]),             # 34.7°N, 极轨发射
    'Jiuquan': np.array([5, 9, 6, 16, 9, 13, 22, 25, 36, 21]),              # 40.9°N, 中国主力
    'Taiyuan': np.array([5, 4, 2, 6, 10, 7, 12, 14, 9, 13]),                # 37.5°N, 极轨
    'Xichang': np.array([9, 7, 8, 17, 13, 13, 16, 16, 15, 19]),             # 28.2°N, GEO发射
    'Wenchang': np.array([0, 2, 2, 0, 1, 5, 5, 6, 4, 9]),                   # 19.6°N, 新建潜力大
}

# 发射基地信息
spaceport_info = {
    'Cape Canaveral': {'cn': '卡纳维拉尔角', 'lat': 28.5, 'country': 'USA'},
    'Kennedy': {'cn': '肯尼迪航天中心', 'lat': 28.5, 'country': 'USA'},
    'Vandenberg': {'cn': '范登堡基地', 'lat': 34.7, 'country': 'USA'},
    'Jiuquan': {'cn': '酒泉', 'lat': 40.9, 'country': 'China'},
    'Taiyuan': {'cn': '太原', 'lat': 37.5, 'country': 'China'},
    'Xichang': {'cn': '西昌', 'lat': 28.2, 'country': 'China'},
    'Wenchang': {'cn': '文昌', 'lat': 19.6, 'country': 'China'},
}

# 容量上限
capacity_limits = {
    'Cape Canaveral': 200,
    'Kennedy': 150,
    'Vandenberg': 150,
    'Jiuquan': 120,
    'Taiyuan': 80,
    'Xichang': 80,
    'Wenchang': 100,
}

# ============================================================================
# 预测模型
# ============================================================================
def logistic_growth_predict(data, years_hist, years_future, K):
    """逻辑增长模型"""
    if len(data) >= 2:
        growth_rates = []
        for i in range(1, len(data)):
            if data[i-1] > 0:
                r = (data[i] - data[i-1]) / max(data[i-1], 1)
                growth_rates.append(r)
        r = np.median(growth_rates) if growth_rates else 0.05
    else:
        r = 0.05
    
    r = np.clip(r, -0.05, 0.15)
    N = data[-1]
    predictions = []
    
    for _ in range(len(years_future)):
        dN = r * N * (1 - N / K)
        N = max(0.1, N + dN)
        predictions.append(N)
    
    return np.array(predictions)


def gm11_predict_bounded(x0, predict_years, K):
    """带边界的GM(1,1)灰色预测"""
    n = len(x0)
    x0_work = np.where(x0 <= 0, 0.1, x0).astype(float)
    
    x1 = np.cumsum(x0_work)
    z1 = (x1[:-1] + x1[1:]) / 2
    B = np.column_stack((-z1, np.ones(n-1)))
    Y = x0_work[1:].reshape(-1, 1)
    
    try:
        params = np.linalg.lstsq(B, Y, rcond=None)[0]
        a, b = params[0, 0], params[1, 0]
    except:
        return np.repeat(x0[-1], predict_years)
    
    def predict_value(k):
        return (x0_work[0] - b/a) * np.exp(-a * k) + b/a
    
    predictions = []
    for k in range(n, n + predict_years):
        x1_pred = predict_value(k)
        x1_prev = predict_value(k - 1)
        x0_pred = x1_pred - x1_prev
        x0_pred = min(max(0, x0_pred), K)
        predictions.append(x0_pred)
    
    return np.array(predictions)


def combined_predict_bounded(years, data, predict_years, K):
    """综合预测"""
    future_years = np.arange(years[-1] + 1, years[-1] + 1 + predict_years)
    
    gm_pred = gm11_predict_bounded(data, predict_years, K)
    logistic_pred = logistic_growth_predict(data, years, future_years, K)
    
    weights_gm = np.linspace(0.6, 0.3, predict_years)
    weights_log = 1 - weights_gm
    
    combined = weights_gm * gm_pred + weights_log * logistic_pred
    combined = np.clip(combined, 0, K)
    
    return combined


# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    predict_start_year = 2025
    predict_end_year = 2060
    predict_years_count = predict_end_year - predict_start_year + 1
    future_years = np.arange(predict_start_year, predict_end_year + 1)

    predictions = {}

    print("=" * 70)
    print("火箭发射基地发射量预测 (优化版 - 7个活跃发射站)")
    print("移除: Baikonur(高纬度), Kourou(发射量骤降), Satish Dhawan(无增长)")
    print("=" * 70)

    for name, data in spaceport_data.items():
        K = capacity_limits[name]
        pred = combined_predict_bounded(years, data, predict_years_count, K)
        predictions[name] = pred
        info = spaceport_info[name]
        idx_2050 = 2050 - predict_start_year
        print(f"{name} ({info['cn']}, {info['lat']}°N): 2050年预测 = {pred[idx_2050]:.1f}")

    # ========================================================================
    # 可视化
    # ========================================================================
    colors = {
        'Cape Canaveral': '#1f77b4',
        'Kennedy': '#ff7f0e',
        'Vandenberg': '#2ca02c',
        'Jiuquan': '#9467bd',
        'Taiyuan': '#8c564b',
        'Xichang': '#e377c2',
        'Wenchang': '#7f7f7f',
    }

    markers = {
        'Cape Canaveral': 'o',
        'Kennedy': 's',
        'Vandenberg': '^',
        'Jiuquan': 'D',
        'Taiyuan': 'v',
        'Xichang': 'p',
        'Wenchang': 'h',
    }

    fig, ax = plt.subplots(figsize=(16, 10), dpi=120)

    for name in spaceport_data.keys():
        info = spaceport_info[name]
        # 历史数据
        ax.plot(years, spaceport_data[name], 
                color=colors[name], 
                linewidth=2.5,
                marker=markers[name],
                markersize=7,
                label=f'{name} ({info["cn"]}, {info["lat"]}°N)')
        
        # 预测数据
        ax.plot(future_years, predictions[name],
                color=colors[name],
                linewidth=1.5,
                linestyle='--',
                alpha=0.7)
        
        # 连接线
        ax.plot([years[-1], future_years[0]], 
                [spaceport_data[name][-1], predictions[name][0]],
                color=colors[name],
                linestyle=':',
                linewidth=1.5,
                alpha=0.5)

    ax.axvline(x=2050, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Year 2050')
    ax.axvline(x=2024.5, color='gray', linestyle='-', linewidth=2, alpha=0.4)

    # 添加注释
    ylim = ax.get_ylim()
    ax.text(2019, 195, 'Historical Data', ha='center', va='top', 
            fontsize=11, color='#333333', fontweight='bold')
    ax.text(2042, 195, 'Predicted Data', ha='center', va='top', 
            fontsize=11, color='#333333', fontweight='bold')

    # 添加关停说明
    ax.text(2015, -15, 'Closed sites: Baikonur (45.6°N, low activity), Kourou (declining), Satish Dhawan (stagnant)', 
            fontsize=9, color='#666666', style='italic')

    ax.set_title('Optimized Spaceport Network Launch Predictions (2015-2060)\n'
                 '7 Active Sites after Closing Inefficient Stations (GM(1,1) + Logistic Growth)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Annual Orbital Launches', fontsize=12)

    ax.legend(loc='upper left', fontsize=9, framealpha=0.95, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='-')
    ax.set_xlim(2014, 2061)
    ax.set_ylim(-20, 220)
    ax.set_xticks(np.arange(2015, 2065, 5))

    plt.tight_layout()

    # 保存图片
    output_dir = r'c:\Users\EDWINJ\Desktop\浙大\竞赛\美赛\比赛期间\results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'spaceport_prediction_7sites.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n图片已保存至: {output_path}")

    # ========================================================================
    # 统计对比
    # ========================================================================
    print("\n" + "=" * 70)
    print("2050年预测发射量排名 (优化后7站)")
    print("=" * 70)

    idx_2050 = 2050 - predict_start_year
    total_2050 = 0

    results_2050 = []
    for name in spaceport_data.keys():
        pred_2050 = predictions[name][idx_2050]
        total_2050 += pred_2050
        info = spaceport_info[name]
        results_2050.append((name, info['cn'], info['lat'], pred_2050))

    results_2050.sort(key=lambda x: x[3], reverse=True)

    print(f"\n{'Rank':<4} {'Spaceport':<18} {'纬度':<8} {'2050预测':<10}")
    print("-" * 50)
    for i, (name, cn_name, lat, pred) in enumerate(results_2050, 1):
        print(f"{i:<4} {name:<18} {lat:>5.1f}°N  {pred:>8.1f}")

    print("-" * 50)
    print(f"{'Total':<4} {'7 Active Sites':<18} {'':8} {total_2050:>8.1f}")

    # 对比原10站
    print("\n" + "=" * 70)
    print("优化效果分析")
    print("=" * 70)
    
    # 原10站2050年总量约707.5，关停3站损失约6.8 (Baikonur 2.5 + Kourou 0.6 + Satish Dhawan 3.7)
    original_total = 707.5
    closed_loss = 6.8
    print(f"原10站2050年预测总量: {original_total:.1f}")
    print(f"关停3站损失量: {closed_loss:.1f} (仅占 {closed_loss/original_total*100:.2f}%)")
    print(f"优化后7站总量: {total_2050:.1f}")
    print(f"保留率: {total_2050/original_total*100:.1f}%")
    
    print("\n关停站点分析:")
    print("  - Baikonur (45.6°N): 纬度最高，发射成本高，发射量持续下降")
    print("  - Kourou (5.2°N): 虽纬度低但发射量从12→3骤降，基础设施老化")
    print("  - Satish Dhawan (13.7°N): 10年无增长，发射能力有限")

    plt.show()
