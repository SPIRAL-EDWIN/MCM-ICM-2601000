# -*- coding: utf-8 -*-
"""
MCM/ICM 2025 - 正确的10个发射场灰色预测分析
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
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互后端
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

#==============================================================================
# 数据收集 - 2015-2024年各发射场年度发射次数
# 数据来源：Wikipedia, SpaceFlightNow, Gunter's Space Page
#==============================================================================

# 年份
years = np.array([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])

# 10个正确发射场数据
spaceport_data = {
    'Alaska': {
        'name': 'Pacific Spaceport Complex, Alaska',
        'latitude': 57.4,  # 高纬度，不利
        'launches': np.array([0, 0, 0, 0, 0, 2, 2, 1, 1, 0]),  # Astra launches mainly, very low
        'country': 'USA'
    },
    'California': {
        'name': 'Vandenberg SFB, California', 
        'latitude': 34.7,
        'launches': np.array([8, 8, 12, 15, 10, 8, 14, 22, 25, 28]),  # SpaceX polar launches
        'country': 'USA'
    },
    'Texas': {
        'name': 'SpaceX Starbase, Texas',
        'latitude': 26.0,  # 较低纬度，有利
        'launches': np.array([0, 0, 0, 0, 0, 0, 0, 0, 3, 5]),  # Starship tests 2023+
        'country': 'USA'
    },
    'Florida': {
        'name': 'Cape Canaveral/KSC, Florida',
        'latitude': 28.5,
        'launches': np.array([18, 18, 29, 31, 21, 31, 45, 57, 72, 85]),  # Main US site
        'country': 'USA'
    },
    'Virginia': {
        'name': 'Wallops/MARS, Virginia',
        'latitude': 37.8,
        'launches': np.array([2, 1, 2, 3, 3, 4, 4, 4, 7, 8]),  # Antares + Electron
        'country': 'USA'
    },
    'Kazakhstan': {
        'name': 'Baikonur Cosmodrome',
        'latitude': 45.6,  # 高纬度
        'launches': np.array([18, 17, 17, 16, 15, 12, 12, 9, 9, 8]),  # Declining
        'country': 'Kazakhstan/Russia'
    },
    'French Guiana': {
        'name': 'Guiana Space Centre (Kourou)',
        'latitude': 5.2,  # 最接近赤道！
        'launches': np.array([12, 11, 11, 11, 9, 5, 6, 5, 4, 3]),  # Declining
        'country': 'France/ESA'
    },
    'India': {
        'name': 'Satish Dhawan Space Centre',
        'latitude': 13.7,
        'launches': np.array([5, 7, 5, 7, 6, 2, 2, 5, 7, 8]),  # Stable, slight growth
        'country': 'India'
    },
    'China-Taiyuan': {
        'name': 'Taiyuan Satellite Launch Center',
        'latitude': 37.5,
        'launches': np.array([6, 8, 7, 11, 13, 10, 14, 12, 14, 15]),  # Polar orbits
        'country': 'China'
    },
    'New Zealand': {
        'name': 'Mahia Peninsula (Rocket Lab)',
        'latitude': -39.3,  # 南半球
        'launches': np.array([0, 0, 1, 3, 6, 6, 6, 9, 10, 16]),  # Rapid growth!
        'country': 'New Zealand'
    }
}

#==============================================================================
# 评估指标计算
#==============================================================================

def calculate_metrics(data, years):
    """计算各发射场的评估指标"""
    metrics = {}
    
    for site, info in data.items():
        launches = info['launches']
        
        # 近10年总发射数
        total = np.sum(launches)
        
        # 2024年发射数
        latest = launches[-1]
        
        # 增长率（复合年均增长率 CAGR，避免除零）
        if launches[0] > 0 and launches[-1] > 0:
            cagr = (launches[-1] / launches[0]) ** (1/9) - 1
        elif launches[-1] > 0:
            # 从0增长，使用近5年数据
            nonzero_idx = np.where(launches > 0)[0]
            if len(nonzero_idx) >= 2:
                start_idx = nonzero_idx[0]
                years_growth = len(launches) - start_idx - 1
                cagr = (launches[-1] / launches[start_idx]) ** (1/max(1, years_growth)) - 1
            else:
                cagr = 0.5  # 新兴发射场，假设50%增长
        else:
            cagr = -0.2  # 停滞，假设-20%
        
        # 近3年趋势（线性回归斜率）
        recent = launches[-3:]
        x = np.arange(3)
        if np.sum(recent) > 0:
            slope = np.polyfit(x, recent, 1)[0]
        else:
            slope = 0
        
        # 纬度评分（越接近赤道越好，0°最优）
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

#==============================================================================
# 综合评分与取舍判断
#==============================================================================

def evaluate_sites(metrics, data):
    """综合评分，判断保留/关闭"""
    scores = {}
    
    # 权重设定
    w_total = 0.25      # 历史总量
    w_latest = 0.20     # 当前活跃度
    w_growth = 0.30     # 增长潜力（CAGR + slope）
    w_latitude = 0.25   # 纬度优势
    
    # 归一化
    totals = [m['total_10y'] for m in metrics.values()]
    latests = [m['latest_2024'] for m in metrics.values()]
    cagrs = [m['cagr'] for m in metrics.values()]
    slopes = [m['recent_slope'] for m in metrics.values()]
    
    max_total = max(totals) if max(totals) > 0 else 1
    max_latest = max(latests) if max(latests) > 0 else 1
    max_cagr = max(cagrs) if max(cagrs) > 0 else 1
    max_slope = max(slopes) if max(slopes) > 0 else 1
    
    for site, m in metrics.items():
        # 归一化评分
        score_total = m['total_10y'] / max_total
        score_latest = m['latest_2024'] / max_latest
        score_cagr = max(0, m['cagr']) / max(max_cagr, 0.01)  # 负增长为0分
        score_slope = max(0, m['recent_slope']) / max(max_slope, 0.01)
        score_growth = 0.6 * score_cagr + 0.4 * score_slope
        score_lat = m['lat_score']
        
        # 综合评分
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
            'recommendation': 'KEEP' if total_score >= 0.20 else 'CLOSE'
        }
    
    return scores

#==============================================================================
# GM(1,1) 灰色预测模型
#==============================================================================

def gm11_predict(x0, n_predict):
    """
    GM(1,1) 灰色预测模型
    x0: 原始数据序列
    n_predict: 预测期数
    """
    n = len(x0)
    
    # 处理零值和负值
    x0 = np.array(x0, dtype=float)
    x0 = np.maximum(x0, 0.1)  # 避免零值
    
    # 累加生成 AGO
    x1 = np.cumsum(x0)
    
    # 紧邻均值生成
    z1 = 0.5 * (x1[:-1] + x1[1:])
    
    # 构建数据矩阵
    B = np.column_stack([-z1, np.ones(n-1)])
    Y = x0[1:].reshape(-1, 1)
    
    # 最小二乘求解
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
    
    # 累减还原
    predictions = np.array(predictions)
    x0_pred = np.diff(predictions, prepend=0)
    x0_pred[0] = x0[0]
    
    return np.maximum(x0_pred, 0)

#==============================================================================
# 带上限约束的预测模型
#==============================================================================

def bounded_prediction(x0, years_future, K=365):
    """
    结合GM(1,1)和逻辑增长的有界预测
    K: 发射次数上限（每年最多365次，即每天1次）
    """
    # GM(1,1)预测
    gm_pred = gm11_predict(x0, years_future)
    
    # 如果数据增长，应用逻辑约束
    if x0[-1] > x0[0]:
        # 估计r（内禀增长率）
        r = np.log(max(x0[-1], 1) / max(x0[0], 0.5)) / len(x0)
        r = np.clip(r, 0.01, 0.5)
        
        # 逻辑增长修正
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
        
        # 混合预测（近期偏GM，远期偏logistic）
        weights_gm = np.exp(-np.arange(len(gm_pred)) * 0.05)
        weights_gm[:len(x0)] = 1
        
        final_pred = weights_gm * gm_pred + (1 - weights_gm) * logistic_pred
        final_pred = np.minimum(final_pred, K)
    else:
        # 下降趋势，直接使用GM预测并设下限
        final_pred = np.maximum(gm_pred, 0)
    
    return final_pred

#==============================================================================
# 主程序
#==============================================================================

if __name__ == '__main__':
    print("="*70)
    print("MCM/ICM 2025 - 正确10发射场评估与灰色预测")
    print("="*70)
    
    # 计算评估指标
    metrics = calculate_metrics(spaceport_data, years)
    
    # 输出评估结果
    print("\n" + "="*70)
    print("发射场评估指标")
    print("="*70)
    print(f"{'Site':<20} {'Lat':>6} {'Total':>6} {'2024':>5} {'CAGR%':>7} {'Slope':>6}")
    print("-"*60)
    
    for site, m in metrics.items():
        print(f"{site:<20} {m['latitude']:>6.1f} {m['total_10y']:>6} "
              f"{m['latest_2024']:>5} {m['cagr']*100:>6.1f}% {m['recent_slope']:>6.1f}")
    
    # 综合评分
    scores = evaluate_sites(metrics, spaceport_data)
    
    print("\n" + "="*70)
    print("综合评分与取舍建议")
    print("="*70)
    print(f"{'Site':<20} {'Score':>7} {'总量':>6} {'活跃':>6} {'增长':>6} {'纬度':>6} {'建议':>8}")
    print("-"*70)
    
    sorted_sites = sorted(scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
    keep_sites = []
    close_sites = []
    
    for site, s in sorted_sites:
        rec = s['recommendation']
        print(f"{site:<20} {s['total_score']:>7.3f} {s['score_total']:>6.2f} "
              f"{s['score_latest']:>6.2f} {s['score_growth']:>6.2f} {s['score_lat']:>6.2f} "
              f"{'✓ '+rec if rec=='KEEP' else '✗ '+rec:>8}")
        
        if rec == 'KEEP':
            keep_sites.append(site)
        else:
            close_sites.append(site)
    
    print("\n" + "="*70)
    print("取舍决策")
    print("="*70)
    print(f"保留发射场 ({len(keep_sites)}个): {', '.join(keep_sites)}")
    print(f"关闭发射场 ({len(close_sites)}个): {', '.join(close_sites)}")
    
    #==========================================================================
    # 灰色预测（保留的发射场）
    #==========================================================================
    
    # 预测到2050年
    years_predict = np.arange(2015, 2051)
    n_history = len(years)
    n_future = len(years_predict) - n_history
    
    predictions = {}
    K = 365  # 每年最多365次发射
    
    for site in keep_sites:
        x0 = spaceport_data[site]['launches']
        pred = bounded_prediction(x0, n_future, K)
        predictions[site] = pred
    
    #==========================================================================
    # 可视化
    #==========================================================================
    
    # 设置图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 颜色方案
    colors = plt.cm.tab10(np.linspace(0, 1, len(keep_sites)))
    color_map = {site: colors[i] for i, site in enumerate(keep_sites)}
    
    # 图1：历史数据与预测
    ax1 = axes[0, 0]
    for site in keep_sites:
        launches = spaceport_data[site]['launches']
        pred = predictions[site]
        
        # 历史数据
        ax1.plot(years, launches, 'o-', color=color_map[site], 
                label=site, markersize=4, linewidth=1.5)
        # 预测数据
        ax1.plot(years_predict[n_history:], pred[n_history:], '--', 
                color=color_map[site], alpha=0.7, linewidth=1.5)
    
    ax1.axvline(x=2024.5, color='gray', linestyle=':', alpha=0.5)
    ax1.axhline(y=K, color='red', linestyle='--', alpha=0.5, label=f'Cap={K}/year')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Annual Launches', fontsize=12)
    ax1.set_title('Spaceport Launch Predictions (GM(1,1) + Logistic Bound)', fontsize=14)
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.set_xlim(2015, 2050)
    ax1.set_ylim(0, K + 20)
    ax1.grid(True, alpha=0.3)
    
    # 图2：2050年预测排名
    ax2 = axes[0, 1]
    pred_2050 = {site: predictions[site][-1] for site in keep_sites}
    sorted_2050 = sorted(pred_2050.items(), key=lambda x: x[1], reverse=True)
    sites_sorted = [x[0] for x in sorted_2050]
    values_sorted = [x[1] for x in sorted_2050]
    colors_sorted = [color_map[s] for s in sites_sorted]
    
    bars = ax2.barh(sites_sorted, values_sorted, color=colors_sorted, alpha=0.8)
    ax2.axvline(x=K, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Predicted Launches in 2050', fontsize=12)
    ax2.set_title('2050 Launch Capacity Ranking', fontsize=14)
    for bar, val in zip(bars, values_sorted):
        ax2.text(val + 5, bar.get_y() + bar.get_height()/2, 
                f'{val:.0f}', va='center', fontsize=10)
    ax2.set_xlim(0, K + 50)
    
    # 图3：纬度 vs 2024发射量（气泡图）
    ax3 = axes[1, 0]
    for site in spaceport_data.keys():
        lat = spaceport_data[site]['latitude']
        launches_2024 = spaceport_data[site]['launches'][-1]
        total = np.sum(spaceport_data[site]['launches'])
        
        color = 'green' if site in keep_sites else 'red'
        marker = 'o' if site in keep_sites else 'x'
        
        ax3.scatter(lat, launches_2024, s=total*5, c=color, alpha=0.6, 
                   marker=marker, label=site if site in ['Florida', 'Alaska'] else '')
        ax3.annotate(site, (lat, launches_2024), fontsize=8, alpha=0.8,
                    xytext=(5, 5), textcoords='offset points')
    
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax3.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Equator')
    ax3.set_xlabel('Latitude (°)', fontsize=12)
    ax3.set_ylabel('Launches in 2024', fontsize=12)
    ax3.set_title('Latitude vs Launch Activity (size=10yr total)', fontsize=14)
    ax3.set_xlim(-60, 70)
    ax3.grid(True, alpha=0.3)
    
    # 图4：增长趋势比较
    ax4 = axes[1, 1]
    x_pos = np.arange(len(spaceport_data))
    site_names = list(spaceport_data.keys())
    
    for i, site in enumerate(site_names):
        m = metrics[site]
        color = 'green' if site in keep_sites else 'red'
        ax4.bar(i, m['cagr']*100, color=color, alpha=0.7)
    
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(site_names, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('CAGR (%)', fontsize=12)
    ax4.set_title('Compound Annual Growth Rate (2015-2024)', fontsize=14)
    ax4.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/spaceport_analysis_correct.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*70)
    print("预测结果 - 保留发射场 (2050年)")
    print("="*70)
    print(f"{'Site':<20} {'2024':>6} {'2030':>6} {'2040':>6} {'2050':>6}")
    print("-"*50)
    
    total_2050 = 0
    for site in keep_sites:
        pred = predictions[site]
        idx_2024 = np.where(years_predict == 2024)[0][0]
        idx_2030 = np.where(years_predict == 2030)[0][0]
        idx_2040 = np.where(years_predict == 2040)[0][0]
        idx_2050 = np.where(years_predict == 2050)[0][0]
        
        print(f"{site:<20} {pred[idx_2024]:>6.0f} {pred[idx_2030]:>6.0f} "
              f"{pred[idx_2040]:>6.0f} {pred[idx_2050]:>6.0f}")
        total_2050 += pred[idx_2050]
    
    print("-"*50)
    print(f"{'Total':<20} {'-':>6} {'-':>6} {'-':>6} {total_2050:>6.0f}")
    print(f"\n年度发射总量上限: {K * len(keep_sites)} 次/年")
    print(f"2050年预测总发射量: {total_2050:.0f} 次/年")
    
    print("\n图像已保存至: results/spaceport_analysis_correct.png")
