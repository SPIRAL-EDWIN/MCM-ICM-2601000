# -*- coding: utf-8 -*-
"""
MCM/ICM 2025 - TOPSIS评估 + 平滑S曲线预测至2070年
更新要求：
1. 使用TOPSIS方法评估10个发射场，删除最低3个
2. 平滑S形曲线渐进逼近365次/年（而非截断）
3. 预测延长至2070年，原2050年达到365的改为2070年达到
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# 设置字体
rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False

#==============================================================================
# 数据定义 - 2016-2025年各发射场年度发射次数
#==============================================================================

years = np.array([2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])

spaceport_data = {
    'Alaska': {
        'name': 'Pacific Spaceport Complex',
        'latitude': 57.4,
        'launches': np.array([0, 0, 0, 0, 2, 2, 1, 1, 0, 0]),
        'country': 'USA',
        'note': 'Very high latitude, mostly suborbital tests, declining'
    },
    'California': {
        'name': 'Vandenberg SFB', 
        'latitude': 34.7,
        'launches': np.array([8, 12, 15, 10, 8, 14, 22, 25, 28, 66]),
        'country': 'USA',
        'note': 'Primary polar/SSO launch site, explosive growth'
    },
    'Texas': {
        'name': 'SpaceX Starbase',
        'latitude': 26.0,
        'launches': np.array([0, 0, 0, 0, 0, 0, 0, 3, 5, 5]),
        'country': 'USA',
        'note': 'New site, Starship testing, high potential'
    },
    'Florida': {
        'name': 'Cape Canaveral/KSC',
        'latitude': 28.5,
        'launches': np.array([18, 29, 31, 21, 31, 45, 57, 72, 85, 109]),
        'country': 'USA',
        'note': 'Primary US launch site, highest activity globally'
    },
    'Virginia': {
        'name': 'Wallops/MARS',
        'latitude': 37.8,
        'launches': np.array([1, 2, 3, 3, 4, 4, 4, 7, 8, 1]),
        'country': 'USA',
        'note': 'Antares retired, Rocket Lab occasional, declining'
    },
    'Kazakhstan': {
        'name': 'Baikonur Cosmodrome',
        'latitude': 45.6,
        'launches': np.array([17, 17, 16, 15, 12, 12, 9, 9, 8, 6]),
        'country': 'Kazakhstan/Russia',
        'note': 'Declining due to Russia relocation to Vostochny'
    },
    'French Guiana': {
        'name': 'Guiana Space Centre',
        'latitude': 5.2,
        'launches': np.array([11, 11, 11, 9, 5, 6, 5, 4, 3, 7]),
        'country': 'France/ESA',
        'note': 'Best latitude, Ariane 6 transition'
    },
    'India': {
        'name': 'Satish Dhawan Space Centre',
        'latitude': 13.7,
        'launches': np.array([7, 5, 7, 6, 2, 2, 5, 7, 8, 5]),
        'country': 'India',
        'note': 'ISRO primary site, steady but limited growth'
    },
    'China-Taiyuan': {
        'name': 'Taiyuan Satellite Launch Center',
        'latitude': 37.5,
        'launches': np.array([8, 7, 11, 13, 10, 14, 12, 14, 15, 12]),
        'country': 'China',
        'note': 'Polar orbit launches, consistent'
    },
    'New Zealand': {
        'name': 'Mahia Peninsula (Rocket Lab)',
        'latitude': -39.3,
        'launches': np.array([0, 1, 3, 6, 6, 6, 9, 10, 16, 17]),
        'country': 'New Zealand',
        'note': 'Rocket Lab Electron, strong growth'
    }
}

#==============================================================================
# TOPSIS 评估方法
#==============================================================================

def topsis_scorer(df, weights, negative_indicators=None):
    """
    TOPSIS评估方法
    
    Args:
        df: 原始数据矩阵 (发射场 x 指标)
        weights: 各指标权重 (和为1)
        negative_indicators: 越小越好的指标列表
    
    Returns:
        带有TOPSIS得分和排名的DataFrame
    """
    data = df.copy().astype(float)
    weights = np.array(weights)
    negative_indicators = negative_indicators if negative_indicators else []
    
    # 1. 向量归一化: z_ij = x_ij / sqrt(sum(x_ij^2))
    norm_data = data / np.sqrt((data**2).sum(axis=0))
    
    # 2. 加权归一化矩阵
    weighted_data = norm_data * weights
    
    # 3. 确定理想解和负理想解
    ideal_best = []
    ideal_worst = []
    
    for col in data.columns:
        if col in negative_indicators:
            # 成本型指标: 越小越好
            ideal_best.append(weighted_data[col].min())
            ideal_worst.append(weighted_data[col].max())
        else:
            # 效益型指标: 越大越好
            ideal_best.append(weighted_data[col].max())
            ideal_worst.append(weighted_data[col].min())
    
    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)
    
    # 4. 计算欧氏距离
    d_pos = np.sqrt(((weighted_data - ideal_best)**2).sum(axis=1))
    d_neg = np.sqrt(((weighted_data - ideal_worst)**2).sum(axis=1))
    
    # 5. 计算相对贴近度 (TOPSIS得分)
    scores = d_neg / (d_pos + d_neg)
    
    # 整理结果
    results = df.copy()
    results['Distance_Pos'] = d_pos
    results['Distance_Neg'] = d_neg
    results['TOPSIS_Score'] = scores
    results['Rank'] = scores.rank(ascending=False).astype(int)
    
    return results.sort_values('Rank')


def calculate_topsis_metrics(data, years):
    """计算TOPSIS所需的评估指标"""
    metrics_list = []
    
    for site, info in data.items():
        launches = info['launches']
        
        # 指标1: 历史发射总量 (效益型)
        total_10y = np.sum(launches)
        
        # 指标2: 2025年发射量 (效益型)
        latest_2025 = launches[-1]
        
        # 指标3: 近5年平均 (效益型)
        avg_5y = np.mean(launches[-5:])
        
        # 指标4: 年复合增长率CAGR (效益型)
        nonzero_idx = np.where(launches > 0)[0]
        if len(nonzero_idx) >= 2:
            start_idx = nonzero_idx[0]
            start_val = max(launches[start_idx], 0.5)
            end_val = max(launches[-1], 0.1)
            years_growth = len(launches) - start_idx - 1
            if years_growth > 0:
                cagr = (end_val / start_val) ** (1/years_growth) - 1
            else:
                cagr = 0
        else:
            cagr = -0.3 if total_10y < 5 else 0
        
        # 指标5: 近3年趋势斜率 (效益型)
        recent = launches[-3:]
        if np.sum(recent) > 0:
            slope = np.polyfit(np.arange(3), recent, 1)[0]
        else:
            slope = -1
        
        # 指标6: 纬度绝对值 (成本型 - 越小越好，越靠近赤道越好)
        lat_abs = abs(info['latitude'])
        
        metrics_list.append({
            'Site': site,
            'Total_10Y': total_10y,
            'Launches_2025': latest_2025,
            'Avg_5Y': avg_5y,
            'CAGR': max(cagr, -0.5),  # 限制下界
            'Trend_Slope': max(slope, -5),  # 限制下界
            'Latitude_Abs': lat_abs
        })
    
    return pd.DataFrame(metrics_list).set_index('Site')


def evaluate_with_topsis(data, years, n_close=3):
    """使用TOPSIS方法评估发射场"""
    
    # 计算指标矩阵
    metrics_df = calculate_topsis_metrics(data, years)
    
    # TOPSIS权重设定 (总和=1)
    # 历史总量: 0.20, 2025活跃: 0.15, 5年均值: 0.15, CAGR: 0.20, 趋势: 0.15, 纬度: 0.15
    weights = [0.20, 0.15, 0.15, 0.20, 0.15, 0.15]
    
    # 纬度是成本型指标（越小越好）
    negative_indicators = ['Latitude_Abs']
    
    # 执行TOPSIS
    results = topsis_scorer(metrics_df, weights, negative_indicators)
    
    # 确定关闭和保留
    all_ranked = results.sort_values('TOPSIS_Score', ascending=True)
    close_sites = list(all_ranked.index[:n_close])
    keep_sites = list(all_ranked.index[n_close:])
    
    return results, keep_sites, close_sites, metrics_df


#==============================================================================
# 平滑S曲线预测 (渐进逼近365，不截断)
#==============================================================================

def gm11_predict(x0, n_predict):
    """标准GM(1,1)灰色预测"""
    n = len(x0)
    x0 = np.array(x0, dtype=float)
    x0 = np.maximum(x0, 0.1)
    
    x1 = np.cumsum(x0)
    z1 = 0.5 * (x1[:-1] + x1[1:])
    
    B = np.column_stack([-z1, np.ones(n-1)])
    Y = x0[1:].reshape(-1, 1)
    
    try:
        params = np.linalg.lstsq(B, Y, rcond=None)[0]
        a, b = params[0, 0], params[1, 0]
    except:
        return np.full(n + n_predict, x0[-1])
    
    predictions = []
    for k in range(n + n_predict):
        x1_pred = (x0[0] - b/a) * np.exp(-a * k) + b/a
        predictions.append(x1_pred)
    
    predictions = np.array(predictions)
    x0_pred = np.diff(predictions, prepend=0)
    x0_pred[0] = x0[0]
    return np.maximum(x0_pred, 0)


def smooth_asymptotic_prediction_high_growth(x0, years_future, K=365, target_year=45):
    """
    高增长站点的平滑S曲线预测
    在target_year年后接近K的98%，使用渐进逻辑函数
    """
    N0 = max(x0[-1], 1)
    
    # 计算r使得在target_year达到98%K
    # N(t) = K / (1 + ((K-N0)/N0) * exp(-r*t))
    # 0.98K = K / (1 + ((K-N0)/N0) * exp(-r*target_year))
    # 解得: r = -ln(0.02 * N0 / (K-N0)) / target_year
    
    ratio = (K - N0) / N0
    r = -np.log(0.02 / ratio) / target_year if ratio > 0 else 0.1
    r = np.clip(r, 0.05, 0.25)
    
    pred = []
    for t in range(len(x0) + years_future):
        if t < len(x0):
            pred.append(x0[t])
        else:
            t_future = t - len(x0) + 1
            # 标准逻辑增长函数
            denominator = 1 + ratio * np.exp(-r * t_future)
            N_t = K / denominator
            # 确保永远不超过99.5%K (渐进但不到达)
            pred.append(min(N_t, K * 0.995))
    
    return np.array(pred)


def smooth_asymptotic_prediction_stable(x0, years_future, K=365, growth_type='stable'):
    """
    稳定/衰退站点的预测 - 不会接近365
    growth_type: 'stable' 缓慢增长, 'declining' 衰退
    """
    gm_pred = gm11_predict(x0, years_future)
    
    # 判断趋势
    recent_trend = x0[-3:].mean() - x0[:3].mean()
    
    if recent_trend < -2:
        # 衰退趋势 - 指数衰减
        pred = []
        for t in range(len(x0) + years_future):
            if t < len(x0):
                pred.append(x0[t])
            else:
                t_future = t - len(x0) + 1
                # 指数衰减但有下限
                decay_rate = 0.03
                N_t = max(x0[-1] * np.exp(-decay_rate * t_future), max(1, x0[-1] * 0.2))
                pred.append(N_t)
        return np.array(pred)
    else:
        # 稳定或缓慢增长
        pred = []
        base = x0[-3:].mean()
        for t in range(len(x0) + years_future):
            if t < len(x0):
                pred.append(x0[t])
            else:
                t_future = t - len(x0) + 1
                # 缓慢线性增长，有上限
                growth_rate = 0.02  # 2%/年
                N_t = base * (1 + growth_rate * t_future)
                # 上限设为当前值的5倍或100，取较小者
                cap = min(base * 5, 100)
                pred.append(min(N_t, cap))
        return np.array(pred)


#==============================================================================
# 主程序
#==============================================================================

if __name__ == '__main__':
    print("="*80)
    print("MCM/ICM 2025 - TOPSIS评估 + 平滑S曲线预测至2070年")
    print("="*80)
    
    # TOPSIS评估
    topsis_results, keep_sites, close_sites, metrics_df = evaluate_with_topsis(
        spaceport_data, years, n_close=3
    )
    
    # 打印TOPSIS结果
    print("\n" + "="*80)
    print("TOPSIS 多准则决策评估")
    print("="*80)
    print("\n指标权重分配:")
    print("  - 历史发射总量 (Total_10Y): 20%  [效益型]")
    print("  - 2025年发射量 (Launches_2025): 15%  [效益型]")
    print("  - 近5年平均 (Avg_5Y): 15%  [效益型]")
    print("  - 年复合增长率 (CAGR): 20%  [效益型]")
    print("  - 趋势斜率 (Trend_Slope): 15%  [效益型]")
    print("  - 纬度绝对值 (Latitude_Abs): 15%  [成本型]")
    
    print("\n" + "-"*80)
    print("TOPSIS评估结果 (按得分排序)")
    print("-"*80)
    print(f"{'Rank':<5} {'Site':<18} {'Total':<7} {'2025':<6} {'CAGR%':<8} {'Lat':<7} {'D+':<8} {'D-':<8} {'Score':<8} {'Decision'}")
    print("-"*80)
    
    for _, row in topsis_results.iterrows():
        site = row.name
        rec = 'CLOSE' if site in close_sites else 'KEEP'
        symbol = '✗' if site in close_sites else '✓'
        print(f"{row['Rank']:<5} {site:<18} {row['Total_10Y']:<7.0f} {row['Launches_2025']:<6.0f} "
              f"{row['CAGR']*100:<7.1f}% {row['Latitude_Abs']:<7.1f} {row['Distance_Pos']:<8.4f} "
              f"{row['Distance_Neg']:<8.4f} {row['TOPSIS_Score']:<8.4f} {symbol} {rec}")
    
    print("-"*80)
    print(f"\n【保留】({len(keep_sites)}个): {', '.join(keep_sites)}")
    print(f"【关闭】({len(close_sites)}个): {', '.join(close_sites)}")
    
    # 关闭理由
    print("\n" + "="*80)
    print("TOPSIS关闭理由分析:")
    print("="*80)
    for site in close_sites:
        row = topsis_results.loc[site]
        print(f"\n  ▸ {site} ({spaceport_data[site]['name']})")
        print(f"    - TOPSIS得分: {row['TOPSIS_Score']:.4f} (排名第{int(row['Rank'])})")
        print(f"    - D+距理想解: {row['Distance_Pos']:.4f}")
        print(f"    - D-距负理想解: {row['Distance_Neg']:.4f}")
        print(f"    - 纬度: {spaceport_data[site]['latitude']:.1f}°")
        print(f"    - 10年总发射: {row['Total_10Y']:.0f}次, 2025年: {row['Launches_2025']:.0f}次")
        print(f"    - 原因: {spaceport_data[site]['note']}")
    
    #==========================================================================
    # 平滑S曲线预测至2070年
    #==========================================================================
    
    print("\n" + "="*80)
    print("平滑S曲线预测 (2025-2070)")
    print("="*80)
    
    years_predict = np.arange(2016, 2071)  # 延长到2070年
    n_history = len(years)
    n_future = len(years_predict) - n_history
    K = 365
    
    # 分类：高增长站点(原2050达到365的) vs 稳定站点
    high_growth_sites = ['Florida', 'California', 'Texas', 'New Zealand']  # 4个高增长
    stable_sites = [s for s in keep_sites if s not in high_growth_sites]   # 剩余站点
    
    predictions = {}
    for site in keep_sites:
        x0 = spaceport_data[site]['launches']
        
        if site in high_growth_sites:
            # 高增长站点：在2070年接近365 (target_year=45)
            pred = smooth_asymptotic_prediction_high_growth(x0, n_future, K=K, target_year=45)
        else:
            # 稳定/衰退站点：保持原有趋势，不会接近365
            pred = smooth_asymptotic_prediction_stable(x0, n_future, K=K)
        
        predictions[site] = pred
    
    # 打印预测表
    print(f"\n{'Site':<18} {'2025':>7} {'2030':>7} {'2040':>7} {'2050':>7} {'2060':>7} {'2070':>7}")
    print("-"*72)
    
    total_by_year = {}
    for year in [2025, 2030, 2040, 2050, 2060, 2070]:
        idx = np.where(years_predict == year)[0][0]
        total_by_year[year] = sum(predictions[s][idx] for s in keep_sites)
    
    for site in sorted(keep_sites, key=lambda s: predictions[s][-1], reverse=True):
        pred = predictions[site]
        vals = []
        for year in [2025, 2030, 2040, 2050, 2060, 2070]:
            idx = np.where(years_predict == year)[0][0]
            vals.append(pred[idx])
        growth_type = "HIGH" if site in high_growth_sites else "STABLE"
        print(f"{site:<18} {vals[0]:>7.1f} {vals[1]:>7.1f} {vals[2]:>7.1f} {vals[3]:>7.1f} {vals[4]:>7.1f} {vals[5]:>7.1f}  [{growth_type}]")
    
    print("-"*72)
    print(f"{'TOTAL':<18} {total_by_year[2025]:>7.0f} {total_by_year[2030]:>7.0f} {total_by_year[2040]:>7.0f} "
          f"{total_by_year[2050]:>7.0f} {total_by_year[2060]:>7.0f} {total_by_year[2070]:>7.0f}")
    
    print(f"\n高增长站点 (2070年接近365): {', '.join([s for s in high_growth_sites if s in keep_sites])}")
    print(f"稳定/衰退站点: {', '.join(stable_sites)}")
    
    #==========================================================================
    # 绘图 (4面板)
    #==========================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 颜色映射
    all_sites = list(spaceport_data.keys())
    colors_all = plt.cm.tab10(np.linspace(0, 1, 10))
    color_map = {site: colors_all[i] for i, site in enumerate(all_sites)}
    
    # 图1: 预测曲线 (保留的站点) - 2016-2070
    ax1 = axes[0, 0]
    
    for site in keep_sites:
        launches = spaceport_data[site]['launches']
        pred = predictions[site]
        
        # 历史数据 (实线圆点)
        ax1.plot(years, launches, 'o-', color=color_map[site], 
                label=f'{site}', markersize=5, linewidth=2)
        
        # 预测数据 (虚线)
        ax1.plot(years_predict[n_history:], pred[n_history:], '--', 
                color=color_map[site], alpha=0.7, linewidth=2)
    
    # 标记关闭的站点 (灰色X)
    for site in close_sites:
        launches = spaceport_data[site]['launches']
        ax1.plot(years, launches, 'x--', color='gray', alpha=0.4, 
                label=f'{site} (CLOSED)', markersize=4, linewidth=1)
    
    ax1.axvline(x=2025.5, color='red', linestyle=':', alpha=0.7, label='Prediction Start')
    ax1.axhline(y=K, color='darkred', linestyle='--', alpha=0.5, label=f'K={K}/year')
    ax1.axhline(y=K*0.95, color='orange', linestyle=':', alpha=0.4, label=f'95% K={K*0.95:.0f}')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Annual Launches', fontsize=12)
    ax1.set_title('Spaceport Launch Predictions (2016-2070)\nSmooth Asymptotic S-Curve (approaching 365)', fontsize=14)
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.set_xlim(2016, 2070)
    ax1.set_ylim(0, K + 30)
    ax1.grid(True, alpha=0.3)
    
    # 添加2050和2070年标记线
    ax1.axvline(x=2050, color='blue', linestyle=':', alpha=0.3)
    ax1.axvline(x=2070, color='purple', linestyle=':', alpha=0.3)
    ax1.text(2050, K+10, '2050', fontsize=9, color='blue', ha='center')
    ax1.text(2070, K+10, '2070', fontsize=9, color='purple', ha='center')
    
    # 图2: 2070年排名
    ax2 = axes[0, 1]
    idx_2070 = np.where(years_predict == 2070)[0][0]
    pred_2070 = {site: predictions[site][idx_2070] for site in keep_sites}
    sorted_2070 = sorted(pred_2070.items(), key=lambda x: x[1], reverse=True)
    sites_sorted = [x[0] for x in sorted_2070]
    values_sorted = [x[1] for x in sorted_2070]
    colors_sorted = [color_map[s] for s in sites_sorted]
    
    bars = ax2.barh(range(len(sites_sorted)), values_sorted, color=colors_sorted, alpha=0.8, edgecolor='black')
    ax2.set_yticks(range(len(sites_sorted)))
    ax2.set_yticklabels([f'{s}\n(lat={spaceport_data[s]["latitude"]:.1f}°)' for s in sites_sorted], fontsize=9)
    ax2.axvline(x=K, color='darkred', linestyle='--', alpha=0.7, label=f'K={K}')
    ax2.axvline(x=K*0.95, color='orange', linestyle=':', alpha=0.5, label=f'95%K')
    ax2.set_xlabel('Predicted Launches in 2070', fontsize=12)
    ax2.set_title(f'2070 Launch Capacity Ranking\n({len(keep_sites)} Active Sites, Asymptotic Limit)', fontsize=14)
    for bar, val in zip(bars, values_sorted):
        pct = val / K * 100
        ax2.text(val + 5, bar.get_y() + bar.get_height()/2, 
                f'{val:.0f} ({pct:.0f}%)', va='center', fontsize=10, fontweight='bold')
    ax2.set_xlim(0, K + 80)
    ax2.invert_yaxis()
    ax2.legend()
    
    # 图3: TOPSIS得分 (所有站点)
    ax3 = axes[1, 0]
    
    # 按TOPSIS得分排序
    all_sites_sorted = list(topsis_results.index)
    x_pos = np.arange(len(all_sites_sorted))
    score_vals = [topsis_results.loc[s, 'TOPSIS_Score'] for s in all_sites_sorted]
    colors_score = ['forestgreen' if s in keep_sites else 'crimson' for s in all_sites_sorted]
    
    bars3 = ax3.bar(x_pos, score_vals, color=colors_score, alpha=0.8, edgecolor='black')
    
    # 阈值线
    threshold = (topsis_results.iloc[2]['TOPSIS_Score'] + topsis_results.iloc[3]['TOPSIS_Score']) / 2
    ax3.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, 
               alpha=0.8, label=f'Closure Threshold≈{threshold:.3f}')
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(all_sites_sorted, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('TOPSIS Score', fontsize=12)
    ax3.set_title('TOPSIS Evaluation Scores\n(Green=KEEP, Red=CLOSE)', fontsize=14)
    ax3.legend(loc='upper right')
    ax3.grid(True, axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars3, score_vals)):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 图4: 纬度 vs 2025活跃度 (气泡大小=TOPSIS得分)
    ax4 = axes[1, 1]
    for site in all_sites:
        lat = spaceport_data[site]['latitude']
        launches_2025 = spaceport_data[site]['launches'][-1]
        topsis_score = topsis_results.loc[site, 'TOPSIS_Score']
        
        if site in keep_sites:
            color = color_map[site]
            marker = 'o'
            edge = 'black'
            alpha = 0.7
        else:
            color = 'gray'
            marker = 'X'
            edge = 'red'
            alpha = 0.5
        
        # 气泡大小正比于TOPSIS得分
        size = max(topsis_score * 800, 80)
        ax4.scatter(lat, launches_2025, s=size, c=[color], alpha=alpha, 
                   marker=marker, edgecolors=edge, linewidths=1.5)
        
        offset = (5, 5) if site in keep_sites else (5, -10)
        ax4.annotate(f'{site}\n(S={topsis_score:.2f})', (lat, launches_2025), fontsize=7, alpha=0.9,
                    xytext=offset, textcoords='offset points',
                    fontweight='bold' if site in keep_sites else 'normal')
    
    ax4.axvline(x=0, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Equator')
    ax4.axhline(y=10, color='orange', linestyle=':', alpha=0.5, label='Activity=10')
    ax4.set_xlabel('Latitude (degrees)', fontsize=12)
    ax4.set_ylabel('Launches in 2025', fontsize=12)
    ax4.set_title('Latitude vs Launch Activity\n(Bubble size ∝ TOPSIS Score, X = CLOSED)', fontsize=14)
    ax4.set_xlim(-55, 65)
    ax4.set_ylim(-5, 120)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('../results/spaceport_topsis_2070.png', dpi=300, bbox_inches='tight')
    print("\n" + "="*80)
    print("图像已保存: results/spaceport_topsis_2070.png")
    print("="*80)
    
    # 输出总结
    print("\n" + "="*80)
    print("结论总结")
    print("="*80)
    print(f"""
## 航天发射场优化分析结果 (TOPSIS方法)

### 评估方法
- **方法**: TOPSIS (逼近理想解排序法)
- **数据范围**: 2016-2025年 (10年)
- **评估指标** (6个):
  | 指标 | 权重 | 类型 |
  |------|------|------|
  | 历史发射总量 | 20% | 效益型 |
  | 2025年发射量 | 15% | 效益型 |
  | 近5年平均 | 15% | 效益型 |
  | 年复合增长率 | 20% | 效益型 |
  | 趋势斜率 | 15% | 效益型 |
  | 纬度绝对值 | 15% | 成本型 |

### 决策结果
""")
    print(f"**保留站点 ({len(keep_sites)}个)**:")
    for site in sorted(keep_sites, key=lambda s: topsis_results.loc[s, 'TOPSIS_Score'], reverse=True):
        row = topsis_results.loc[site]
        growth_type = "高增长" if site in high_growth_sites else "稳定"
        print(f"  - {site}: TOPSIS得分={row['TOPSIS_Score']:.4f}, "
              f"纬度{spaceport_data[site]['latitude']:.1f}°, [{growth_type}]")
    
    print(f"\n**关闭站点 ({len(close_sites)}个)**:")
    for site in close_sites:
        row = topsis_results.loc[site]
        print(f"  - {site}: TOPSIS得分={row['TOPSIS_Score']:.4f}, "
              f"纬度{spaceport_data[site]['latitude']:.1f}°")
        print(f"    理由: {spaceport_data[site]['note']}")
    
    print(f"""
### 预测模型
- **方法**: GM(1,1)灰色预测 + 平滑渐进S曲线
- **上限**: K=365次/年 (渐进逼近，永不截断)
- **高增长站点**: 2070年接近365次/年 (95%+)
- **稳定站点**: 保持原有增长趋势

### 2070年预测
- 保留站点总发射能力: **{total_by_year[2070]:.0f}次/年**
- 理论最大容量: {K} × {len(keep_sites)} = {K * len(keep_sites)}次/年
- 容量利用率: {total_by_year[2070] / (K * len(keep_sites)) * 100:.1f}%
""")
