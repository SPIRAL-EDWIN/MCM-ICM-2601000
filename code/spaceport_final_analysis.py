# -*- coding: utf-8 -*-
"""
MCM/ICM 2025 - 正确的10个发射场灰色预测分析 (最终版)
题目要求的10个基站：
1. Alaska (Pacific Spaceport Complex) - 美国
2. California (Vandenberg SFB) - 美国  
3. Texas (SpaceX Starbase) - 美国
4. Florida (Cape Canaveral/KSC) - 美国
5. Virginia (Wallops/MARS) - 美国
6. Kazakhstan (Baikonur Cosmodrome)
7. French Guiana (Guiana Space Centre/Kourou)
8. Satish Dhawan Space Centre (India)
9. Taiyuan Satellite Launch Center (China)
10. Mahia Peninsula (New Zealand) - Rocket Lab

发射上限: 每年365次 (即每天最多1次)
"""

import numpy as np
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
# 数据定义 - 2015-2024年各发射场年度发射次数
#==============================================================================

years = np.array([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])

spaceport_data = {
    'Alaska': {
        'name': 'Pacific Spaceport Complex',
        'latitude': 57.4,
        'launches': np.array([0, 0, 0, 0, 0, 2, 2, 1, 1, 0]),
        'country': 'USA',
        'note': 'Very high latitude, mostly suborbital tests'
    },
    'California': {
        'name': 'Vandenberg SFB', 
        'latitude': 34.7,
        'launches': np.array([8, 8, 12, 15, 10, 8, 14, 22, 25, 28]),
        'country': 'USA',
        'note': 'Primary polar/SSO launch site'
    },
    'Texas': {
        'name': 'SpaceX Starbase',
        'latitude': 26.0,
        'launches': np.array([0, 0, 0, 0, 0, 0, 0, 0, 3, 5]),
        'country': 'USA',
        'note': 'New site, Starship testing, rapid growth expected'
    },
    'Florida': {
        'name': 'Cape Canaveral/KSC',
        'latitude': 28.5,
        'launches': np.array([18, 18, 29, 31, 21, 31, 45, 57, 72, 85]),
        'country': 'USA',
        'note': 'Primary US launch site, high activity'
    },
    'Virginia': {
        'name': 'Wallops/MARS',
        'latitude': 37.8,
        'launches': np.array([2, 1, 2, 3, 3, 4, 4, 4, 7, 8]),
        'country': 'USA',
        'note': 'Antares + Rocket Lab Electron'
    },
    'Kazakhstan': {
        'name': 'Baikonur Cosmodrome',
        'latitude': 45.6,
        'launches': np.array([18, 17, 17, 16, 15, 12, 12, 9, 9, 8]),
        'country': 'Kazakhstan/Russia',
        'note': 'Declining due to Russia relocation to Vostochny'
    },
    'French Guiana': {
        'name': 'Guiana Space Centre',
        'latitude': 5.2,
        'launches': np.array([12, 11, 11, 11, 9, 5, 6, 5, 4, 3]),
        'country': 'France/ESA',
        'note': 'Best latitude but declining, Ariane 6 transition'
    },
    'India': {
        'name': 'Satish Dhawan Space Centre',
        'latitude': 13.7,
        'launches': np.array([5, 7, 5, 7, 6, 2, 2, 5, 7, 8]),
        'country': 'India',
        'note': 'ISRO primary site, steady growth'
    },
    'China-Taiyuan': {
        'name': 'Taiyuan Satellite Launch Center',
        'latitude': 37.5,
        'launches': np.array([6, 8, 7, 11, 13, 10, 14, 12, 14, 15]),
        'country': 'China',
        'note': 'Polar orbit launches, consistent growth'
    },
    'New Zealand': {
        'name': 'Mahia Peninsula (Rocket Lab)',
        'latitude': -39.3,
        'launches': np.array([0, 0, 1, 3, 6, 6, 6, 9, 10, 16]),
        'country': 'New Zealand',
        'note': 'Rocket Lab Electron, rapid growth'
    }
}

#==============================================================================
# 评估函数
#==============================================================================

def calculate_metrics(data, years):
    metrics = {}
    for site, info in data.items():
        launches = info['launches']
        total = np.sum(launches)
        latest = launches[-1]
        
        # CAGR计算
        if launches[0] > 0 and launches[-1] > 0:
            cagr = (launches[-1] / launches[0]) ** (1/9) - 1
        elif launches[-1] > 0:
            nonzero_idx = np.where(launches > 0)[0]
            if len(nonzero_idx) >= 2:
                start_idx = nonzero_idx[0]
                years_growth = len(launches) - start_idx - 1
                cagr = (launches[-1] / launches[start_idx]) ** (1/max(1, years_growth)) - 1
            else:
                cagr = 0.5
        else:
            cagr = -0.2
        
        # 近3年趋势
        recent = launches[-3:]
        slope = np.polyfit(np.arange(3), recent, 1)[0] if np.sum(recent) > 0 else 0
        
        # 纬度评分
        lat_score = 1 - abs(info['latitude']) / 90
        
        metrics[site] = {
            'total_10y': total,
            'latest_2024': latest,
            'cagr': cagr,
            'recent_slope': slope,
            'latitude': info['latitude'],
            'lat_score': lat_score
        }
    return metrics

def evaluate_sites(metrics):
    scores = {}
    w_total, w_latest, w_growth, w_latitude = 0.25, 0.20, 0.30, 0.25
    
    totals = [m['total_10y'] for m in metrics.values()]
    latests = [m['latest_2024'] for m in metrics.values()]
    cagrs = [m['cagr'] for m in metrics.values()]
    slopes = [m['recent_slope'] for m in metrics.values()]
    
    max_total = max(totals) if max(totals) > 0 else 1
    max_latest = max(latests) if max(latests) > 0 else 1
    max_cagr = max(cagrs) if max(cagrs) > 0 else 1
    max_slope = max(slopes) if max(slopes) > 0 else 1
    
    for site, m in metrics.items():
        score_total = m['total_10y'] / max_total
        score_latest = m['latest_2024'] / max_latest
        score_cagr = max(0, m['cagr']) / max(max_cagr, 0.01)
        score_slope = max(0, m['recent_slope']) / max(max_slope, 0.01)
        score_growth = 0.6 * score_cagr + 0.4 * score_slope
        score_lat = m['lat_score']
        
        total_score = (w_total * score_total + w_latest * score_latest + 
                      w_growth * score_growth + w_latitude * score_lat)
        
        scores[site] = {
            'total_score': total_score,
            'score_total': score_total,
            'score_latest': score_latest,
            'score_growth': score_growth,
            'score_lat': score_lat,
            'recommendation': 'KEEP' if total_score >= 0.15 else 'CLOSE'
        }
    return scores

#==============================================================================
# GM(1,1) 灰色预测
#==============================================================================

def gm11_predict(x0, n_predict):
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

def bounded_prediction(x0, years_future, K=365):
    """带逻辑增长约束的预测"""
    gm_pred = gm11_predict(x0, years_future)
    
    if x0[-1] > x0[0] and x0[-1] > 0:
        # 估计增长率
        r = np.log(max(x0[-1], 1) / max(x0[0], 0.5)) / len(x0)
        r = np.clip(r, 0.01, 0.5)
        N0 = max(x0[-1], 1)
        
        logistic_pred = []
        for t in range(len(gm_pred)):
            if t < len(x0):
                logistic_pred.append(x0[t])
            else:
                t_future = t - len(x0) + 1
                N_t = K / (1 + ((K - N0) / N0) * np.exp(-r * t_future))
                logistic_pred.append(N_t)
        logistic_pred = np.array(logistic_pred)
        
        # 混合权重
        weights_gm = np.exp(-np.arange(len(gm_pred)) * 0.05)
        weights_gm[:len(x0)] = 1
        final_pred = weights_gm * gm_pred + (1 - weights_gm) * logistic_pred
        final_pred = np.minimum(final_pred, K)
    else:
        # 下降趋势
        final_pred = np.maximum(gm_pred, 0)
    
    return final_pred

#==============================================================================
# 主程序
#==============================================================================

if __name__ == '__main__':
    print("="*70)
    print("MCM/ICM 2025 - 正确10发射场评估与灰色预测")
    print("="*70)
    
    # 计算指标
    metrics = calculate_metrics(spaceport_data, years)
    scores = evaluate_sites(metrics)
    
    # 打印评估结果
    print("\n" + "-"*70)
    print("发射场综合评估 (权重: 历史25%, 活跃20%, 增长30%, 纬度25%)")
    print("-"*70)
    print(f"{'Site':<18} {'Lat':>6} {'Total':>6} {'2024':>5} {'CAGR%':>7} {'Score':>6} {'Decision':>10}")
    print("-"*70)
    
    sorted_sites = sorted(scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
    keep_sites = []
    close_sites = []
    
    for site, s in sorted_sites:
        m = metrics[site]
        rec = s['recommendation']
        symbol = "✓" if rec == 'KEEP' else "✗"
        print(f"{site:<18} {m['latitude']:>6.1f} {m['total_10y']:>6} {m['latest_2024']:>5} "
              f"{m['cagr']*100:>6.1f}% {s['total_score']:>6.3f} {symbol} {rec:>8}")
        if rec == 'KEEP':
            keep_sites.append(site)
        else:
            close_sites.append(site)
    
    print("-"*70)
    print(f"\n【保留】({len(keep_sites)}个): {', '.join(keep_sites)}")
    print(f"【关闭】({len(close_sites)}个): {', '.join(close_sites)}")
    
    # 关闭理由
    if close_sites:
        print("\n关闭理由:")
        for site in close_sites:
            m = metrics[site]
            print(f"  • {site}: 纬度{m['latitude']:.1f}°, "
                  f"10年总{m['total_10y']}次, 2024年{m['latest_2024']}次, "
                  f"CAGR={m['cagr']*100:.1f}%")
    
    # 预测
    years_predict = np.arange(2015, 2051)
    n_history = len(years)
    n_future = len(years_predict) - n_history
    K = 365  # 每年最多365次
    
    predictions = {}
    for site in keep_sites:
        x0 = spaceport_data[site]['launches']
        pred = bounded_prediction(x0, n_future, K)
        predictions[site] = pred
    
    #==========================================================================
    # 绘图
    #==========================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = plt.cm.tab10(np.linspace(0, 1, len(keep_sites)))
    color_map = {site: colors[i] for i, site in enumerate(keep_sites)}
    
    # 图1: 预测曲线
    ax1 = axes[0, 0]
    for site in keep_sites:
        launches = spaceport_data[site]['launches']
        pred = predictions[site]
        ax1.plot(years, launches, 'o-', color=color_map[site], 
                label=site, markersize=5, linewidth=2)
        ax1.plot(years_predict[n_history:], pred[n_history:], '--', 
                color=color_map[site], alpha=0.7, linewidth=2)
    
    ax1.axvline(x=2024.5, color='gray', linestyle=':', alpha=0.5, label='Prediction Start')
    ax1.axhline(y=K, color='red', linestyle='--', alpha=0.5, label=f'Max={K}/year')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Annual Launches', fontsize=12)
    ax1.set_title('Spaceport Launch Predictions (GM(1,1) + Logistic Bound)', fontsize=14)
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.set_xlim(2015, 2050)
    ax1.set_ylim(0, K + 30)
    ax1.grid(True, alpha=0.3)
    
    # 图2: 2050年排名
    ax2 = axes[0, 1]
    idx_2050 = np.where(years_predict == 2050)[0][0]
    pred_2050 = {site: predictions[site][idx_2050] for site in keep_sites}
    sorted_2050 = sorted(pred_2050.items(), key=lambda x: x[1], reverse=True)
    sites_sorted = [x[0] for x in sorted_2050]
    values_sorted = [x[1] for x in sorted_2050]
    colors_sorted = [color_map[s] for s in sites_sorted]
    
    bars = ax2.barh(range(len(sites_sorted)), values_sorted, color=colors_sorted, alpha=0.8)
    ax2.set_yticks(range(len(sites_sorted)))
    ax2.set_yticklabels(sites_sorted)
    ax2.axvline(x=K, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Predicted Launches in 2050', fontsize=12)
    ax2.set_title('2050 Launch Capacity Ranking (K=365)', fontsize=14)
    for bar, val in zip(bars, values_sorted):
        ax2.text(val + 5, bar.get_y() + bar.get_height()/2, 
                f'{val:.0f}', va='center', fontsize=10)
    ax2.set_xlim(0, K + 60)
    ax2.invert_yaxis()
    
    # 图3: 综合评分
    ax3 = axes[1, 0]
    all_sites = list(spaceport_data.keys())
    x_pos = np.arange(len(all_sites))
    score_vals = [scores[s]['total_score'] for s in all_sites]
    colors_score = ['green' if s in keep_sites else 'red' for s in all_sites]
    
    bars3 = ax3.bar(x_pos, score_vals, color=colors_score, alpha=0.7)
    ax3.axhline(y=0.15, color='orange', linestyle='--', alpha=0.8, label='Threshold=0.15')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(all_sites, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Composite Score', fontsize=12)
    ax3.set_title('Site Evaluation Scores (Green=KEEP, Red=CLOSE)', fontsize=14)
    ax3.legend()
    ax3.grid(True, axis='y', alpha=0.3)
    
    # 图4: 纬度vs活跃度
    ax4 = axes[1, 1]
    for site in all_sites:
        lat = spaceport_data[site]['latitude']
        launches_2024 = spaceport_data[site]['launches'][-1]
        total = np.sum(spaceport_data[site]['launches'])
        
        color = 'green' if site in keep_sites else 'red'
        marker = 'o' if site in keep_sites else 'x'
        size = max(total * 3, 50)
        
        ax4.scatter(lat, launches_2024, s=size, c=color, alpha=0.6, marker=marker)
        ax4.annotate(site, (lat, launches_2024), fontsize=8, alpha=0.8,
                    xytext=(5, 5), textcoords='offset points')
    
    ax4.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Equator')
    ax4.set_xlabel('Latitude (degrees)', fontsize=12)
    ax4.set_ylabel('Launches in 2024', fontsize=12)
    ax4.set_title('Latitude vs Launch Activity (bubble size = 10yr total)', fontsize=14)
    ax4.set_xlim(-60, 70)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('../results/spaceport_final_analysis.png', dpi=300, bbox_inches='tight')
    print("\n图像已保存: results/spaceport_final_analysis.png")
    
    # 输出预测结果
    print("\n" + "="*70)
    print("灰色预测结果 (K=365次/年)")
    print("="*70)
    print(f"{'Site':<18} {'2024':>6} {'2030':>6} {'2040':>6} {'2050':>6}")
    print("-"*50)
    
    total_2050 = 0
    for site in keep_sites:
        pred = predictions[site]
        idx_2024 = np.where(years_predict == 2024)[0][0]
        idx_2030 = np.where(years_predict == 2030)[0][0]
        idx_2040 = np.where(years_predict == 2040)[0][0]
        
        print(f"{site:<18} {pred[idx_2024]:>6.0f} {pred[idx_2030]:>6.0f} "
              f"{pred[idx_2040]:>6.0f} {pred[idx_2050]:>6.0f}")
        total_2050 += pred[idx_2050]
    
    print("-"*50)
    print(f"{'Total':<18} {'-':>6} {'-':>6} {'-':>6} {total_2050:>6.0f}")
    
    print(f"\n理论最大发射量: {K} × {len(keep_sites)} = {K * len(keep_sites)} 次/年")
    print(f"2050年预测总量: {total_2050:.0f} 次/年")
    print(f"容量利用率: {total_2050 / (K * len(keep_sites)) * 100:.1f}%")
