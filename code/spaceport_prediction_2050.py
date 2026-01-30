# -*- coding: utf-8 -*-
"""
火箭发射基地发射量预测 (2050年及之后)
使用改进的GM(1,1)灰色预测模型 + 逻辑增长约束
10个主要全球火箭发射基地

作者: MCM/ICM 建模团队
日期: 2025年1月
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 历史数据 (2015-2024年) - 来源: Wikipedia "Year in spaceflight"
# ============================================================================
years = np.array([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])

# 10个主要发射基地的历史发射数据
spaceport_data = {
    'Cape Canaveral': np.array([17, 18, 7, 17, 13, 20, 19, 38, 59, 67]),
    'Kennedy': np.array([0, 0, 12, 3, 3, 10, 12, 19, 13, 26]),
    'Vandenberg': np.array([7, 3, 9, 9, 7, 5, 10, 16, 30, 47]),
    'Baikonur': np.array([18, 11, 13, 9, 13, 7, 14, 7, 9, 8]),
    'Jiuquan': np.array([5, 9, 6, 16, 9, 13, 22, 25, 36, 21]),
    'Taiyuan': np.array([5, 4, 2, 6, 10, 7, 12, 14, 9, 13]),
    'Xichang': np.array([9, 7, 8, 17, 13, 13, 16, 16, 15, 19]),
    'Wenchang': np.array([0, 2, 2, 0, 1, 5, 5, 6, 4, 9]),
    'Satish Dhawan': np.array([5, 7, 5, 7, 6, 2, 2, 5, 7, 5]),
    'Kourou': np.array([12, 11, 11, 11, 9, 7, 7, 6, 3, 3]),
}

# 发射基地的中英文名称映射
spaceport_names_cn = {
    'Cape Canaveral': '卡纳维拉尔角',
    'Kennedy': '肯尼迪航天中心',
    'Vandenberg': '范登堡基地',
    'Baikonur': '拜科努尔',
    'Jiuquan': '酒泉',
    'Taiyuan': '太原',
    'Xichang': '西昌',
    'Wenchang': '文昌',
    'Satish Dhawan': '萨迪什·达万',
    'Kourou': '库鲁'
}

# 每个发射基地的容量上限 (考虑基础设施扩展潜力)
capacity_limits = {
    'Cape Canaveral': 200,  # SpaceX主要基地，潜力巨大
    'Kennedy': 150,         # 主要NASA基地
    'Vandenberg': 150,      # 极轨发射主要基地
    'Baikonur': 50,         # 老化设施，增长受限
    'Jiuquan': 120,         # 中国主要发射场
    'Taiyuan': 80,          # 极轨卫星发射
    'Xichang': 80,          # 地球同步轨道发射
    'Wenchang': 100,        # 新建设施，发展潜力大
    'Satish Dhawan': 50,    # 印度主要发射场
    'Kourou': 40,           # 欧空局主要发射场，受限
}

# ============================================================================
# 预测模型函数
# ============================================================================
def logistic_growth_predict(data, years_hist, years_future, K):
    """
    逻辑增长模型预测
    dN/dt = r*N*(1 - N/K)
    """
    # 估计增长率 r
    if len(data) >= 2:
        growth_rates = []
        for i in range(1, len(data)):
            if data[i-1] > 0:
                r = (data[i] - data[i-1]) / max(data[i-1], 1)
                growth_rates.append(r)
        r = np.median(growth_rates) if growth_rates else 0.05
    else:
        r = 0.05
    
    # 限制增长率在合理范围
    r = np.clip(r, -0.05, 0.15)
    
    # 从最后一个数据点开始预测
    N = data[-1]
    predictions = []
    
    for _ in range(len(years_future)):
        dN = r * N * (1 - N / K)
        N = max(0.1, N + dN)
        predictions.append(N)
    
    return np.array(predictions)


def gm11_predict_bounded(x0, predict_years, K):
    """
    带边界约束的GM(1,1)灰色预测模型
    """
    n = len(x0)
    x0_work = np.where(x0 <= 0, 0.1, x0).astype(float)
    
    # 1-AGO累加
    x1 = np.cumsum(x0_work)
    
    # 构建矩阵
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
    """
    综合预测：GM(1,1) + 逻辑增长，带容量约束
    """
    future_years = np.arange(years[-1] + 1, years[-1] + 1 + predict_years)
    
    gm_pred = gm11_predict_bounded(data, predict_years, K)
    logistic_pred = logistic_growth_predict(data, years, future_years, K)
    
    # 加权平均 (近期更信任GM，远期更信任逻辑增长)
    weights_gm = np.linspace(0.6, 0.3, predict_years)
    weights_log = 1 - weights_gm
    
    combined = weights_gm * gm_pred + weights_log * logistic_pred
    combined = np.clip(combined, 0, K)
    
    return combined


# ============================================================================
# 主预测程序
# ============================================================================
if __name__ == "__main__":
    predict_start_year = 2025
    predict_end_year = 2060
    predict_years_count = predict_end_year - predict_start_year + 1
    future_years = np.arange(predict_start_year, predict_end_year + 1)

    predictions = {}

    print("=" * 60)
    print("火箭发射基地发射量预测 (2025-2060)")
    print("=" * 60)

    for name, data in spaceport_data.items():
        K = capacity_limits[name]
        pred = combined_predict_bounded(years, data, predict_years_count, K)
        predictions[name] = pred
        idx_2050 = 2050 - predict_start_year
        print(f"{name} ({spaceport_names_cn[name]}): 2050年预测发射量 = {pred[idx_2050]:.1f}")

    # ========================================================================
    # 可视化
    # ========================================================================
    colors = {
        'Cape Canaveral': '#1f77b4',
        'Kennedy': '#ff7f0e',
        'Vandenberg': '#2ca02c',
        'Baikonur': '#d62728',
        'Jiuquan': '#9467bd',
        'Taiyuan': '#8c564b',
        'Xichang': '#e377c2',
        'Wenchang': '#7f7f7f',
        'Satish Dhawan': '#bcbd22',
        'Kourou': '#17becf'
    }

    linestyles = {
        'Cape Canaveral': '-',
        'Kennedy': '--',
        'Vandenberg': '-.',
        'Baikonur': ':',
        'Jiuquan': '-',
        'Taiyuan': '--',
        'Xichang': '-.',
        'Wenchang': ':',
        'Satish Dhawan': '-',
        'Kourou': '--'
    }

    fig, ax = plt.subplots(figsize=(16, 10), dpi=120)

    for name in spaceport_data.keys():
        ax.plot(years, spaceport_data[name], 
                color=colors[name], 
                linestyle=linestyles[name],
                linewidth=2.5,
                marker='o',
                markersize=6,
                label=f'{name} ({spaceport_names_cn[name]})')
        
        ax.plot(future_years, predictions[name],
                color=colors[name],
                linestyle=linestyles[name],
                linewidth=1.5,
                alpha=0.7)
        
        ax.plot([years[-1], future_years[0]], 
                [spaceport_data[name][-1], predictions[name][0]],
                color=colors[name],
                linestyle=':',
                linewidth=1.5,
                alpha=0.5)

    ax.axvline(x=2050, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Year 2050')
    ax.axvline(x=2024.5, color='gray', linestyle='-', linewidth=2, alpha=0.4)

    ylim = ax.get_ylim()
    ax.text(2019, ylim[1]*0.95, 'Historical Data', ha='center', va='top', 
            fontsize=11, color='#333333', fontweight='bold')
    ax.text(2042, ylim[1]*0.95, 'Predicted Data', ha='center', va='top', 
            fontsize=11, color='#333333', fontweight='bold')

    ax.set_title('Global Major Spaceport Launch Predictions (2015-2060)\n'
                 'Based on GM(1,1) Grey Prediction Model with Logistic Growth Constraint', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Annual Orbital Launches', fontsize=12)

    ax.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.95, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='-')
    ax.set_xlim(2014, 2061)
    ax.set_ylim(0, 220)
    ax.set_xticks(np.arange(2015, 2065, 5))

    plt.tight_layout()

    # 保存图片
    output_dir = r'c:\Users\EDWINJ\Desktop\浙大\竞赛\美赛\比赛期间\results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'spaceport_prediction_2050.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n图片已保存至: {output_path}")

    # ========================================================================
    # 2050年详细预测
    # ========================================================================
    print("\n" + "=" * 60)
    print("2050年各发射基地预测发射量排名")
    print("=" * 60)

    idx_2050 = 2050 - predict_start_year
    total_2050 = 0

    results_2050 = []
    for name in spaceport_data.keys():
        pred_2050 = predictions[name][idx_2050]
        total_2050 += pred_2050
        results_2050.append((name, spaceport_names_cn[name], pred_2050))

    results_2050.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{'Rank':<4} {'Spaceport':<20} {'Name (CN)':<15} {'2050 Predicted':<15}")
    print("-" * 60)
    for i, (name, cn_name, pred) in enumerate(results_2050, 1):
        print(f"{i:<4} {name:<20} {cn_name:<15} {pred:>10.1f}")

    print("-" * 60)
    print(f"{'Total':<4} {'10 Spaceports':<20} {'':<15} {total_2050:>10.1f}")

    # ========================================================================
    # 趋势分析
    # ========================================================================
    print("\n" + "=" * 60)
    print("趋势分析 (2024年 → 2050年)")
    print("=" * 60)

    for name, cn_name, pred_2050 in results_2050:
        base_2024 = spaceport_data[name][-1]
        change = pred_2050 - base_2024
        change_pct = (change / max(base_2024, 1)) * 100
        
        if change_pct > 20:
            trend = "↑ Strong Growth"
        elif change_pct > 0:
            trend = "↗ Moderate Growth"
        elif change_pct > -20:
            trend = "→ Stable"
        else:
            trend = "↓ Decline"
        
        print(f"{name}: {base_2024:.0f} → {pred_2050:.1f} ({change_pct:+.1f}%) {trend}")

    plt.show()
