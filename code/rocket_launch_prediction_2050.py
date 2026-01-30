# -*- coding: utf-8 -*-
"""
火箭发射数量预测 - 2050年预测
使用多种模型进行预测：
1. 指数增长模型
2. Logistic增长模型 (考虑承载能力)
3. 灰色预测GM(1,1)
4. 多项式回归
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

# 历史数据 (轨道发射总数)
# 来源: Wikipedia - Timeline of spaceflight
historical_data = {
    # 2010年代
    2010: 74,
    2011: 84,
    2012: 78,
    2013: 81,
    2014: 92,
    2015: 87,
    2016: 85,
    2017: 90,
    2018: 114,
    2019: 103,
    # 2020年代
    2020: 114,
    2021: 145,
    2022: 186,
    2023: 223,
    2024: 261,
    2025: 290,  # 估计值 (基于2024趋势)
}

years = np.array(list(historical_data.keys()))
launches = np.array(list(historical_data.values()))

# 预测年份
future_years = np.arange(2026, 2051)
all_years = np.concatenate([years, future_years])

print("=" * 60)
print("🚀 火箭发射数量预测分析 - 预测2050年")
print("=" * 60)
print(f"\n历史数据 ({years[0]}-{years[-1]}):")
for y, l in historical_data.items():
    print(f"  {y}: {l} 次发射")

# ============================================================
# 模型1: 指数增长模型 L(t) = a * exp(r * t)
# ============================================================
def exponential_model(t, a, r):
    return a * np.exp(r * t)

# 归一化时间
t_normalized = years - years[0]

try:
    popt_exp, _ = curve_fit(exponential_model, t_normalized, launches, p0=[70, 0.1], maxfev=5000)
    a_exp, r_exp = popt_exp
    
    future_t = all_years - years[0]
    pred_exp = exponential_model(future_t, a_exp, r_exp)
    
    print(f"\n📈 模型1: 指数增长模型")
    print(f"   参数: a={a_exp:.2f}, r={r_exp:.4f} (年增长率≈{r_exp*100:.2f}%)")
    print(f"   2050年预测: {int(pred_exp[-1])} 次发射")
except Exception as e:
    print(f"指数模型拟合失败: {e}")
    pred_exp = None

# ============================================================
# 模型2: Logistic增长模型 (考虑资源限制)
# L(t) = K / (1 + exp(-r*(t-t0)))
# ============================================================
def logistic_model(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))

try:
    # 初始猜测: K=1500 (假设每年最多发射1500次), r=0.3, t0=10
    popt_log, _ = curve_fit(logistic_model, t_normalized, launches, 
                            p0=[1500, 0.3, 15], maxfev=10000,
                            bounds=([500, 0.05, 5], [10000, 1.0, 50]))
    K_log, r_log, t0_log = popt_log
    
    pred_log = logistic_model(future_t, K_log, r_log, t0_log)
    
    print(f"\n📊 模型2: Logistic增长模型 (有承载能力上限)")
    print(f"   参数: K={K_log:.0f} (承载能力), r={r_log:.4f}, t0={t0_log:.2f}")
    print(f"   2050年预测: {int(pred_log[-1])} 次发射")
except Exception as e:
    print(f"Logistic模型拟合失败: {e}")
    pred_log = None

# ============================================================
# 模型3: 灰色预测GM(1,1)
# ============================================================
def grey_model_gm11(data):
    """GM(1,1)灰色预测模型"""
    n = len(data)
    x0 = np.array(data)
    
    # 一次累加生成序列
    x1 = np.cumsum(x0)
    
    # 紧邻均值生成序列
    z1 = 0.5 * (x1[:-1] + x1[1:])
    
    # 构建矩阵
    B = np.column_stack([-z1, np.ones(n-1)])
    Y = x0[1:].reshape(-1, 1)
    
    # 最小二乘求解
    u = np.linalg.lstsq(B, Y, rcond=None)[0]
    a, b = u.flatten()
    
    # 预测公式
    def predict(k):
        x1_pred = (x0[0] - b/a) * np.exp(-a * k) + b/a
        if k == 0:
            return x0[0]
        x1_pred_prev = (x0[0] - b/a) * np.exp(-a * (k-1)) + b/a
        return x1_pred - x1_pred_prev
    
    return predict, a, b

try:
    # 使用近年数据进行灰色预测 (2018-2025)
    recent_data = list(historical_data.values())[-8:]
    gm_predict, gm_a, gm_b = grey_model_gm11(recent_data)
    
    # 预测2026-2050
    pred_gm = []
    for i, y in enumerate(all_years):
        if y <= years[-1]:
            idx = list(historical_data.keys()).index(y) - (len(historical_data) - 8)
            if idx >= 0:
                pred_gm.append(gm_predict(idx))
            else:
                pred_gm.append(historical_data[y])
        else:
            pred_gm.append(gm_predict(8 + (y - 2026)))
    
    pred_gm = np.array(pred_gm)
    
    print(f"\n📉 模型3: 灰色预测GM(1,1)")
    print(f"   参数: a={gm_a:.4f}, b={gm_b:.2f}")
    print(f"   2050年预测: {int(pred_gm[-1])} 次发射")
except Exception as e:
    print(f"灰色预测模型失败: {e}")
    pred_gm = None

# ============================================================
# 模型4: 多项式回归 (2次多项式)
# ============================================================
try:
    coeffs = np.polyfit(t_normalized, launches, 2)
    poly = np.poly1d(coeffs)
    pred_poly = poly(future_t)
    
    print(f"\n📐 模型4: 二次多项式回归")
    print(f"   参数: {coeffs[0]:.4f}t² + {coeffs[1]:.4f}t + {coeffs[2]:.4f}")
    print(f"   2050年预测: {int(pred_poly[-1])} 次发射")
except Exception as e:
    print(f"多项式回归失败: {e}")
    pred_poly = None

# ============================================================
# 综合预测 - 加权平均
# ============================================================
print("\n" + "=" * 60)
print("🎯 综合预测结果")
print("=" * 60)

# 计算各模型在历史数据上的RMSE
def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

models = {}
if pred_exp is not None:
    rmse_exp = rmse(launches, exponential_model(t_normalized, a_exp, r_exp))
    models['指数增长'] = {'pred_2050': int(pred_exp[-1]), 'rmse': rmse_exp, 'predictions': pred_exp}
    
if pred_log is not None:
    rmse_log = rmse(launches, logistic_model(t_normalized, K_log, r_log, t0_log))
    models['Logistic'] = {'pred_2050': int(pred_log[-1]), 'rmse': rmse_log, 'predictions': pred_log}

if pred_poly is not None:
    rmse_poly = rmse(launches, poly(t_normalized))
    models['多项式'] = {'pred_2050': int(pred_poly[-1]), 'rmse': rmse_poly, 'predictions': pred_poly}

# 根据RMSE计算权重 (RMSE越小权重越大)
total_inv_rmse = sum(1/m['rmse'] for m in models.values())
weights = {name: (1/m['rmse'])/total_inv_rmse for name, m in models.items()}

weighted_pred_2050 = sum(weights[name] * m['pred_2050'] for name, m in models.items())

print("\n各模型2050年预测对比:")
print("-" * 50)
for name, m in models.items():
    print(f"  {name}: {m['pred_2050']:,} 次发射 (RMSE={m['rmse']:.2f}, 权重={weights[name]:.3f})")

print("-" * 50)
print(f"\n✨ 加权综合预测 2050年: {int(weighted_pred_2050):,} 次火箭发射")

# ============================================================
# 情景分析
# ============================================================
print("\n" + "=" * 60)
print("📊 情景分析")
print("=" * 60)

print("\n🔸 保守情景 (Logistic模型 - 考虑资源限制):")
if pred_log is not None:
    print(f"   2030年: {int(logistic_model(2030-years[0], K_log, r_log, t0_log)):,} 次")
    print(f"   2040年: {int(logistic_model(2040-years[0], K_log, r_log, t0_log)):,} 次")
    print(f"   2050年: {int(logistic_model(2050-years[0], K_log, r_log, t0_log)):,} 次")

print("\n🔸 乐观情景 (指数增长模型):")
if pred_exp is not None:
    print(f"   2030年: {int(exponential_model(2030-years[0], a_exp, r_exp)):,} 次")
    print(f"   2040年: {int(exponential_model(2040-years[0], a_exp, r_exp)):,} 次")
    print(f"   2050年: {int(exponential_model(2050-years[0], a_exp, r_exp)):,} 次")

print("\n🔸 中等情景 (多项式模型):")
if pred_poly is not None:
    print(f"   2030年: {int(poly(2030-years[0])):,} 次")
    print(f"   2040年: {int(poly(2040-years[0])):,} 次")
    print(f"   2050年: {int(poly(2050-years[0])):,} 次")

# ============================================================
# 关键驱动因素分析
# ============================================================
print("\n" + "=" * 60)
print("🔍 关键驱动因素分析")
print("=" * 60)
print("""
增长驱动因素:
1. SpaceX Starship可重复使用 → 发射成本降至$2M/次 → 大幅增加发射频率
2. 卫星互联网星座 (Starlink, Kuiper, Guowang) → 需要数千次发射
3. 中国商业航天崛起 → 每年新增50-100次发射
4. 太空旅游市场成熟 → 每年可能新增100+次发射
5. 月球/火星任务 → 2030年代后每年10-30次深空发射

限制因素:
1. 发射场容量限制 (全球约20个活跃发射场)
2. 轨道频谱资源有限 (ITU频率分配)
3. 太空碎片风险 (Kessler效应)
4. 环境监管加强
5. 地缘政治因素
""")

# ============================================================
# 可视化
# ============================================================
plt.figure(figsize=(14, 8))
plt.style.use('seaborn-v0_8-whitegrid')

# 历史数据
plt.scatter(years, launches, color='black', s=100, zorder=5, label='历史数据', marker='o')

# 各模型预测
if pred_exp is not None:
    plt.plot(all_years, pred_exp, 'r--', linewidth=2, alpha=0.7, label=f'指数增长 (2050: {int(pred_exp[-1]):,})')
if pred_log is not None:
    plt.plot(all_years, pred_log, 'b-', linewidth=2, alpha=0.7, label=f'Logistic (2050: {int(pred_log[-1]):,})')
if pred_poly is not None:
    plt.plot(all_years, pred_poly, 'g-.', linewidth=2, alpha=0.7, label=f'多项式 (2050: {int(pred_poly[-1]):,})')

# 加权综合预测线
weighted_predictions = []
for i in range(len(all_years)):
    wp = sum(weights[name] * m['predictions'][i] for name, m in models.items())
    weighted_predictions.append(wp)
plt.plot(all_years, weighted_predictions, 'm-', linewidth=3, alpha=0.9, 
         label=f'加权综合 (2050: {int(weighted_pred_2050):,})')

# 标注关键年份
key_years = [2030, 2040, 2050]
for ky in key_years:
    idx = np.where(all_years == ky)[0][0]
    plt.axvline(x=ky, color='gray', linestyle=':', alpha=0.5)
    plt.annotate(f'{ky}', xy=(ky, plt.ylim()[0]), xytext=(ky-1, plt.ylim()[0]+50),
                fontsize=10, alpha=0.7)

plt.xlabel('年份', fontsize=12)
plt.ylabel('年度火箭发射总数', fontsize=12)
plt.title('全球火箭发射数量预测 (2010-2050)', fontsize=14, fontweight='bold')
plt.legend(loc='upper left', fontsize=10)
plt.xlim(2010, 2052)
plt.ylim(0, max(max(pred_exp) if pred_exp is not None else 0, 
                max(pred_log) if pred_log is not None else 0,
                max(pred_poly) if pred_poly is not None else 0) * 1.1)

plt.tight_layout()
plt.savefig('c:/Users/EDWINJ/Desktop/浙大/竞赛/美赛/比赛期间/results/rocket_launch_prediction_2050.png', 
            dpi=300, bbox_inches='tight')
print("\n📊 预测图表已保存至: results/rocket_launch_prediction_2050.png")

# 最终结论
print("\n" + "=" * 60)
print("🏁 最终结论")
print("=" * 60)
print(f"""
基于多模型综合分析:

📅 2050年全球火箭发射总数预测: {int(weighted_pred_2050):,} 次

预测区间:
- 保守估计 (Logistic): {int(pred_log[-1]) if pred_log is not None else 'N/A':,} 次
- 中等估计 (综合加权): {int(weighted_pred_2050):,} 次  
- 乐观估计 (指数增长): {int(pred_exp[-1]) if pred_exp is not None else 'N/A':,} 次

关键假设:
1. 可重复使用火箭技术成熟 (SpaceX, Blue Origin, 中国长征系列)
2. 商业航天持续快速发展
3. 无重大地缘政治冲突影响航天活动
4. 太空碎片问题得到有效管理

年均增长率: 约{((weighted_pred_2050/261)**(1/26)-1)*100:.1f}% (2024-2050)
""")

plt.show()
