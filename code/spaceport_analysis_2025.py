# -*- coding: utf-8 -*-
"""
MCM/ICM 2025 - 2016-2025年数据的发射场灰色预测分析 (更新版)
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
本版本：使用2016-2025年数据，关闭3个发射场
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
# 数据定义 - 2016-2025年各发射场年度发射次数 (更新版)
#==============================================================================

years = np.array([2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])

spaceport_data = {
    'Alaska': {
        'name': 'Pacific Spaceport Complex',
        'latitude': 57.4,
        # 2016-2025: 几乎无轨道发射活动
        'launches': np.array([0, 0, 0, 0, 2, 2, 1, 1, 0, 0]),
        'country': 'USA',
        'note': 'Very high latitude, mostly suborbital tests, declining'
    },
    'California': {
        'name': 'Vandenberg SFB', 
        'latitude': 34.7,
        # 2016-2025: SpaceX Falcon 9极地轨道+军事发射，2025年猛增
        'launches': np.array([8, 12, 15, 10, 8, 14, 22, 25, 28, 66]),
        'country': 'USA',
        'note': 'Primary polar/SSO launch site, explosive growth'
    },
    'Texas': {
        'name': 'SpaceX Starbase',
        'latitude': 26.0,
        # 2016-2025: 2023开始有Starship测试发射
        'launches': np.array([0, 0, 0, 0, 0, 0, 0, 3, 5, 5]),
        'country': 'USA',
        'note': 'New site, Starship testing, high failure rate in 2025'
    },
    'Florida': {
        'name': 'Cape Canaveral/KSC',
        'latitude': 28.5,
        # 2016-2025: SpaceX + ULA + others, 2025突破100次
        'launches': np.array([18, 29, 31, 21, 31, 45, 57, 72, 85, 109]),
        'country': 'USA',
        'note': 'Primary US launch site, highest activity globally'
    },
    'Virginia': {
        'name': 'Wallops/MARS',
        'latitude': 37.8,
        # 2016-2025: Antares中断后下降，Rocket Lab偶有使用
        'launches': np.array([1, 2, 3, 3, 4, 4, 4, 7, 8, 1]),
        'country': 'USA',
        'note': 'Antares retired, Rocket Lab occasional, declining'
    },
    'Kazakhstan': {
        'name': 'Baikonur Cosmodrome',
        'latitude': 45.6,
        # 2016-2025: 俄罗斯迁往东方航天发射场，持续下降
        'launches': np.array([17, 17, 16, 15, 12, 12, 9, 9, 8, 6]),
        'country': 'Kazakhstan/Russia',
        'note': 'Declining due to Russia relocation to Vostochny'
    },
    'French Guiana': {
        'name': 'Guiana Space Centre',
        'latitude': 5.2,
        # 2016-2025: Ariane 5退役，Ariane 6过渡期
        'launches': np.array([11, 11, 11, 9, 5, 6, 5, 4, 3, 7]),
        'country': 'France/ESA',
        'note': 'Best latitude, Ariane 6 transition, slight recovery'
    },
    'India': {
        'name': 'Satish Dhawan Space Centre',
        'latitude': 13.7,
        # 2016-2025: ISRO稳定发展
        'launches': np.array([7, 5, 7, 6, 2, 2, 5, 7, 8, 5]),
        'country': 'India',
        'note': 'ISRO primary site, steady but limited growth'
    },
    'China-Taiyuan': {
        'name': 'Taiyuan Satellite Launch Center',
        'latitude': 37.5,
        # 2016-2025: 中国航天稳步增长
        'launches': np.array([8, 7, 11, 13, 10, 14, 12, 14, 15, 12]),
        'country': 'China',
        'note': 'Polar orbit launches, consistent'
    },
    'New Zealand': {
        'name': 'Mahia Peninsula (Rocket Lab)',
        'latitude': -39.3,
        # 2016-2025: Rocket Lab Electron快速增长
        'launches': np.array([0, 1, 3, 6, 6, 6, 9, 10, 16, 17]),
        'country': 'New Zealand',
        'note': 'Rocket Lab Electron, strong growth'
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
        
        # CAGR计算 - 考虑起点为0的情况
        nonzero_idx = np.where(launches > 0)[0]
        if len(nonzero_idx) >= 2:
            start_idx = nonzero_idx[0]
            start_val = launches[start_idx]
            end_val = launches[-1]
            years_growth = len(launches) - start_idx - 1
            if end_val > 0 and start_val > 0 and years_growth > 0:
                cagr = (end_val / start_val) ** (1/years_growth) - 1
            else:
                cagr = 0
        else:
            cagr = -0.3 if total < 5 else 0
        
        # 近3年趋势 (slope)
        recent = launches[-3:]
        if np.sum(recent) > 0:
            slope = np.polyfit(np.arange(3), recent, 1)[0]
        else:
            slope = -1
        
        # 近5年平均
        avg_5y = np.mean(launches[-5:])
        
        # 纬度评分 (离赤道越近越好)
        lat_score = 1 - abs(info['latitude']) / 90
        
        metrics[site] = {
            'total_10y': total,
            'latest_2025': latest,
            'avg_5y': avg_5y,
            'cagr': cagr,
            'recent_slope': slope,
            'latitude': info['latitude'],
            'lat_score': lat_score
        }
    return metrics

def evaluate_sites(metrics, n_close=3):
    """
    评估发射场并确定关闭哪些
    权重: 历史总量25%, 最近活跃度20%, 增长潜力30%, 纬度优势25%
    """
    scores = {}
    w_total, w_latest, w_growth, w_latitude = 0.25, 0.20, 0.30, 0.25
    
    # 计算归一化因子
    totals = [m['total_10y'] for m in metrics.values()]
    latests = [m['latest_2025'] for m in metrics.values()]
    avgs = [m['avg_5y'] for m in metrics.values()]
    cagrs = [m['cagr'] for m in metrics.values()]
    slopes = [m['recent_slope'] for m in metrics.values()]
    
    max_total = max(totals) if max(totals) > 0 else 1
    max_latest = max(latests) if max(latests) > 0 else 1
    max_cagr = max(cagrs) if max(cagrs) > 0 else 0.01
    max_slope = max(slopes) if max(slopes) > 0 else 0.01
    
    for site, m in metrics.items():
        # 各维度归一化分数
        score_total = m['total_10y'] / max_total
        score_latest = m['latest_2025'] / max_latest
        
        # 增长分数 (CAGR + 近期斜率)
        score_cagr = max(0, m['cagr']) / max(max_cagr, 0.01)
        score_slope = max(0, m['recent_slope']) / max(max_slope, 0.01)
        score_growth = 0.6 * score_cagr + 0.4 * score_slope
        
        # 纬度分数
        score_lat = m['lat_score']
        
        # 综合分数
        total_score = (w_total * score_total + 
                      w_latest * score_latest + 
                      w_growth * score_growth + 
                      w_latitude * score_lat)
        
        scores[site] = {
            'total_score': total_score,
            'score_total': score_total,
            'score_latest': score_latest,
            'score_growth': score_growth,
            'score_lat': score_lat,
            'components': {
                'historical': w_total * score_total,
                'activity': w_latest * score_latest,
                'growth': w_growth * score_growth,
                'latitude': w_latitude * score_lat
            }
        }
    
    # 按分数排序，最低的n_close个关闭
    sorted_sites = sorted(scores.items(), key=lambda x: x[1]['total_score'])
    close_list = [s[0] for s in sorted_sites[:n_close]]
    keep_list = [s[0] for s in sorted_sites[n_close:]]
    
    for site in scores:
        scores[site]['recommendation'] = 'CLOSE' if site in close_list else 'KEEP'
    
    return scores, keep_list, close_list

#==============================================================================
# GM(1,1) 灰色预测模型
#==============================================================================

def gm11_predict(x0, n_predict):
    """标准GM(1,1)灰色预测"""
    n = len(x0)
    x0 = np.array(x0, dtype=float)
    x0 = np.maximum(x0, 0.1)  # 避免零值
    
    # 一次累加序列
    x1 = np.cumsum(x0)
    
    # 紧邻均值生成序列
    z1 = 0.5 * (x1[:-1] + x1[1:])
    
    # 构建矩阵
    B = np.column_stack([-z1, np.ones(n-1)])
    Y = x0[1:].reshape(-1, 1)
    
    try:
        params = np.linalg.lstsq(B, Y, rcond=None)[0]
        a, b = params[0, 0], params[1, 0]
    except:
        return np.full(n + n_predict, x0[-1])
    
    # 预测
    predictions = []
    for k in range(n + n_predict):
        x1_pred = (x0[0] - b/a) * np.exp(-a * k) + b/a
        predictions.append(x1_pred)
    
    predictions = np.array(predictions)
    x0_pred = np.diff(predictions, prepend=0)
    x0_pred[0] = x0[0]
    return np.maximum(x0_pred, 0)

def bounded_prediction(x0, years_future, K=365):
    """带逻辑增长约束的GM(1,1)预测"""
    gm_pred = gm11_predict(x0, years_future)
    
    # 判断增长趋势
    recent_trend = x0[-3:].mean() - x0[:3].mean()
    
    if recent_trend > 0 and x0[-1] > 0:
        # 上升趋势：使用逻辑增长约束
        # 估计增长率
        nonzero = x0[x0 > 0]
        if len(nonzero) >= 2:
            r = np.log(max(nonzero[-1], 1) / max(nonzero[0], 0.5)) / len(nonzero)
            r = np.clip(r, 0.02, 0.5)
        else:
            r = 0.1
        
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
        
        # 加权混合：近期偏GM，远期偏逻辑
        weights_gm = np.exp(-np.arange(len(gm_pred)) * 0.03)
        weights_gm[:len(x0)] = 1
        final_pred = weights_gm * gm_pred + (1 - weights_gm) * logistic_pred
        final_pred = np.minimum(final_pred, K)
    else:
        # 下降或稳定趋势
        final_pred = np.maximum(gm_pred, 0)
        # 防止无限下降
        final_pred = np.maximum(final_pred, 0)
    
    return final_pred

#==============================================================================
# 主程序
#==============================================================================

if __name__ == '__main__':
    print("="*75)
    print("MCM/ICM 2025 - 10发射场评估与灰色预测 (2016-2025数据)")
    print("目标：关闭3个发射场")
    print("="*75)
    
    # 计算指标
    metrics = calculate_metrics(spaceport_data, years)
    
    # 评估并确定关闭3个
    scores, keep_sites, close_sites = evaluate_sites(metrics, n_close=3)
    
    # 打印评估结果
    print("\n" + "-"*75)
    print("发射场综合评估 (权重: 历史25%, 活跃20%, 增长30%, 纬度25%)")
    print("-"*75)
    print(f"{'Site':<18} {'Lat':>6} {'Total':>6} {'2025':>5} {'Avg5y':>6} {'CAGR%':>7} {'Score':>6} {'Decision':>10}")
    print("-"*75)
    
    sorted_sites = sorted(scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
    
    for site, s in sorted_sites:
        m = metrics[site]
        rec = s['recommendation']
        symbol = "✓" if rec == 'KEEP' else "✗"
        print(f"{site:<18} {m['latitude']:>6.1f} {m['total_10y']:>6} {m['latest_2025']:>5} "
              f"{m['avg_5y']:>6.1f} {m['cagr']*100:>6.1f}% {s['total_score']:>6.3f} {symbol} {rec:>8}")
    
    print("-"*75)
    print(f"\n【保留】({len(keep_sites)}个): {', '.join(keep_sites)}")
    print(f"【关闭】({len(close_sites)}个): {', '.join(close_sites)}")
    
    # 关闭理由
    print("\n" + "="*75)
    print("关闭理由分析:")
    print("="*75)
    for site in close_sites:
        m = metrics[site]
        s = scores[site]
        print(f"\n  ▸ {site} ({spaceport_data[site]['name']})")
        print(f"    - 纬度: {m['latitude']:.1f}° (偏高，发射效率低)")
        print(f"    - 10年总发射: {m['total_10y']}次")
        print(f"    - 2025年发射: {m['latest_2025']}次")
        print(f"    - CAGR: {m['cagr']*100:.1f}%")
        print(f"    - 综合得分: {s['total_score']:.3f} (排名倒数)")
        print(f"    - 原因: {spaceport_data[site]['note']}")
    
    # 灰色预测
    print("\n" + "="*75)
    print("GM(1,1) + 逻辑增长预测 (K=365次/年)")
    print("="*75)
    
    years_predict = np.arange(2016, 2051)
    n_history = len(years)
    n_future = len(years_predict) - n_history
    K = 365
    
    predictions = {}
    for site in keep_sites:
        x0 = spaceport_data[site]['launches']
        pred = bounded_prediction(x0, n_future, K)
        predictions[site] = pred
    
    # 打印预测
    print(f"\n{'Site':<18} {'2025':>7} {'2030':>7} {'2035':>7} {'2040':>7} {'2045':>7} {'2050':>7}")
    print("-"*65)
    
    total_by_year = {}
    for year in [2025, 2030, 2035, 2040, 2045, 2050]:
        idx = np.where(years_predict == year)[0][0]
        total_by_year[year] = sum(predictions[s][idx] for s in keep_sites)
    
    for site in sorted(keep_sites, key=lambda s: predictions[s][-1], reverse=True):
        pred = predictions[site]
        vals = []
        for year in [2025, 2030, 2035, 2040, 2045, 2050]:
            idx = np.where(years_predict == year)[0][0]
            vals.append(pred[idx])
        print(f"{site:<18} {vals[0]:>7.0f} {vals[1]:>7.0f} {vals[2]:>7.0f} {vals[3]:>7.0f} {vals[4]:>7.0f} {vals[5]:>7.0f}")
    
    print("-"*65)
    print(f"{'TOTAL':<18} {total_by_year[2025]:>7.0f} {total_by_year[2030]:>7.0f} {total_by_year[2035]:>7.0f} "
          f"{total_by_year[2040]:>7.0f} {total_by_year[2045]:>7.0f} {total_by_year[2050]:>7.0f}")
    
    print(f"\n理论最大容量: {K} × {len(keep_sites)} = {K * len(keep_sites)} 次/年")
    print(f"2050年预测总量: {total_by_year[2050]:.0f} 次/年")
    print(f"容量利用率: {total_by_year[2050] / (K * len(keep_sites)) * 100:.1f}%")
    
    #==========================================================================
    # 绘图 (4面板)
    #==========================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # 颜色映射
    all_sites = list(spaceport_data.keys())
    colors_all = plt.cm.tab10(np.linspace(0, 1, 10))
    color_map = {site: colors_all[i] for i, site in enumerate(all_sites)}
    
    # 图1: 预测曲线 (保留的站点)
    ax1 = axes[0, 0]
    for site in keep_sites:
        launches = spaceport_data[site]['launches']
        pred = predictions[site]
        ax1.plot(years, launches, 'o-', color=color_map[site], 
                label=f'{site}', markersize=5, linewidth=2)
        ax1.plot(years_predict[n_history:], pred[n_history:], '--', 
                color=color_map[site], alpha=0.7, linewidth=2)
    
    # 标记关闭的站点 (灰色)
    for site in close_sites:
        launches = spaceport_data[site]['launches']
        ax1.plot(years, launches, 'x--', color='gray', alpha=0.4, 
                label=f'{site} (CLOSED)', markersize=4, linewidth=1)
    
    ax1.axvline(x=2025.5, color='red', linestyle=':', alpha=0.7, label='Prediction Start')
    ax1.axhline(y=K, color='darkred', linestyle='--', alpha=0.5, label=f'K={K}/year')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Annual Launches', fontsize=12)
    ax1.set_title('Spaceport Launch Predictions (2016-2050)\nGM(1,1) + Logistic Growth Bound', fontsize=14)
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.set_xlim(2016, 2050)
    ax1.set_ylim(0, K + 50)
    ax1.grid(True, alpha=0.3)
    
    # 图2: 2050年排名
    ax2 = axes[0, 1]
    idx_2050 = np.where(years_predict == 2050)[0][0]
    pred_2050 = {site: predictions[site][idx_2050] for site in keep_sites}
    sorted_2050 = sorted(pred_2050.items(), key=lambda x: x[1], reverse=True)
    sites_sorted = [x[0] for x in sorted_2050]
    values_sorted = [x[1] for x in sorted_2050]
    colors_sorted = [color_map[s] for s in sites_sorted]
    
    bars = ax2.barh(range(len(sites_sorted)), values_sorted, color=colors_sorted, alpha=0.8, edgecolor='black')
    ax2.set_yticks(range(len(sites_sorted)))
    ax2.set_yticklabels([f'{s}\n(lat={spaceport_data[s]["latitude"]:.1f}°)' for s in sites_sorted], fontsize=9)
    ax2.axvline(x=K, color='darkred', linestyle='--', alpha=0.7, label=f'K={K}')
    ax2.set_xlabel('Predicted Launches in 2050', fontsize=12)
    ax2.set_title(f'2050 Launch Capacity Ranking\n({len(keep_sites)} Active Sites)', fontsize=14)
    for bar, val in zip(bars, values_sorted):
        ax2.text(val + 5, bar.get_y() + bar.get_height()/2, 
                f'{val:.0f}', va='center', fontsize=10, fontweight='bold')
    ax2.set_xlim(0, K + 80)
    ax2.invert_yaxis()
    ax2.legend()
    
    # 图3: 综合评分 (所有站点)
    ax3 = axes[1, 0]
    x_pos = np.arange(len(all_sites))
    score_vals = [scores[s]['total_score'] for s in all_sites]
    colors_score = ['forestgreen' if s in keep_sites else 'crimson' for s in all_sites]
    
    bars3 = ax3.bar(x_pos, score_vals, color=colors_score, alpha=0.8, edgecolor='black')
    
    # 添加阈值线 (第3低和第4低之间)
    sorted_scores = sorted(score_vals)
    threshold = (sorted_scores[2] + sorted_scores[3]) / 2
    ax3.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, 
               alpha=0.8, label=f'Closure Threshold≈{threshold:.2f}')
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(all_sites, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Composite Score', fontsize=12)
    ax3.set_title('Site Evaluation Scores\n(Green=KEEP, Red=CLOSE)', fontsize=14)
    ax3.legend(loc='upper right')
    ax3.grid(True, axis='y', alpha=0.3)
    
    # 在柱子上标注分数
    for i, (bar, val) in enumerate(zip(bars3, score_vals)):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.01, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 图4: 纬度 vs 2025活跃度 (气泡大小=10年总量)
    ax4 = axes[1, 1]
    for site in all_sites:
        lat = spaceport_data[site]['latitude']
        launches_2025 = spaceport_data[site]['launches'][-1]
        total = np.sum(spaceport_data[site]['launches'])
        
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
        
        size = max(total * 5, 80)
        ax4.scatter(lat, launches_2025, s=size, c=[color], alpha=alpha, 
                   marker=marker, edgecolors=edge, linewidths=1.5)
        
        # 标注站点名
        offset = (5, 5) if site in keep_sites else (5, -10)
        ax4.annotate(site, (lat, launches_2025), fontsize=8, alpha=0.9,
                    xytext=offset, textcoords='offset points',
                    fontweight='bold' if site in keep_sites else 'normal')
    
    ax4.axvline(x=0, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Equator')
    ax4.axhline(y=10, color='orange', linestyle=':', alpha=0.5, label='Activity=10')
    ax4.set_xlabel('Latitude (degrees)', fontsize=12)
    ax4.set_ylabel('Launches in 2025', fontsize=12)
    ax4.set_title('Latitude vs Launch Activity\n(Bubble size = 10-year total, X = CLOSED)', fontsize=14)
    ax4.set_xlim(-55, 65)
    ax4.set_ylim(-5, 120)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('../results/spaceport_analysis_2025.png', dpi=300, bbox_inches='tight')
    print("\n" + "="*75)
    print("图像已保存: results/spaceport_analysis_2025.png")
    print("="*75)
    
    # 输出Markdown格式的总结
    print("\n" + "="*75)
    print("结论总结 (Markdown格式)")
    print("="*75)
    print("""
## 航天发射场优化分析结果

### 评估方法
- **数据范围**: 2016-2025年 (10年)
- **评估维度**: 
  - 历史发射总量 (25%)
  - 2025年活跃度 (20%)
  - 增长潜力 (CAGR + 趋势斜率) (30%)
  - 纬度优势 (距赤道距离) (25%)
- **预测模型**: GM(1,1)灰色预测 + 逻辑增长上限 (K=365次/年)

### 决策结果
""")
    print(f"**保留站点 ({len(keep_sites)}个)**:")
    for i, site in enumerate(sorted(keep_sites, key=lambda s: scores[s]['total_score'], reverse=True), 1):
        m = metrics[site]
        print(f"  {i}. {site} ({spaceport_data[site]['country']}) - "
              f"纬度{m['latitude']:.1f}°, 2025年{m['latest_2025']}次")
    
    print(f"\n**关闭站点 ({len(close_sites)}个)**:")
    for i, site in enumerate(close_sites, 1):
        m = metrics[site]
        print(f"  {i}. {site} ({spaceport_data[site]['country']}) - "
              f"纬度{m['latitude']:.1f}°, 2025年{m['latest_2025']}次")
        print(f"     理由: {spaceport_data[site]['note']}")
    
    print(f"""
### 2050年预测
- 保留站点总发射能力: **{total_by_year[2050]:.0f}次/年**
- 理论最大容量: {K} × {len(keep_sites)} = {K * len(keep_sites)}次/年
- 容量利用率: {total_by_year[2050] / (K * len(keep_sites)) * 100:.1f}%
""")
