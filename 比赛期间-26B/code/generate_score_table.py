"""
生成决策综合得分表格 (CSV)
- 行: α 从 0% 到 100%，分度值 1%
- 列: w_E 从 0% 到 50%，分度值 1%
- 值: 综合决策得分 S(α, w_E)
"""

import numpy as np
import pandas as pd
from math import exp, sqrt, log10

# =============================================================================
#                           基础参数设置
# =============================================================================

# 总运输量
M = 10**8  # 吨

# 电梯参数
E_E_base = 179000 * 3     # 电梯基础年运输能力 (吨/年)
C_E_cost = 2.2 * 10**5    # 电梯边际成本 (USD/吨)
F_E = 1.2 * 10**8         # 电梯固定成本 (USD/年)
p_E = 0.03                # 电梯年故障概率
sigma_swing = 1.5         # 缆绳摆动参数 (度)
theta_limit = 4.0         # 摆动角度限制 (度)
val_rep_E = 5 * 10**9     # 电梯维修费用 (USD)

# 火箭参数
P_avg = 125               # 火箭平均载荷 (吨/次)
f_avg = 1472              # 年发射频率 (次/年)
E_R = f_avg * P_avg       # 火箭年运输能力 (吨/年)
C_R_cost = 1.175 * 10**6  # 火箭边际成本 (USD/吨)
F_R = 5 * 10**7           # 火箭固定成本 (USD/年)
q_R = 0.95                # 火箭发射成功概率
val_R = 7.5 * 10**6       # 单次发射成本 (USD)

# 电梯有效运输能力
ff = 1 - exp(-theta_limit**2 / (2 * sigma_swing**2))
E_E = E_E_base * ff
N_rocket = f_avg

# 环境指标参数
A_E = 50 * 3      # 电梯用地 (km²)
A_R = 75 * 7      # 火箭用地 (km²)
LDN_E = 62        # 电梯噪音 (dB)
N = 1472
SEL_single = 120
n_daily = N / 365
day_fraction = 0.7
night_fraction = 0.3
n_effective = n_daily * (day_fraction + 10 * night_fraction)
LDN_R = SEL_single + 10 * log10(n_effective) - 49.4

Pol_E = 32        # 电梯排放指标 (常规电网)
Pol_R = 225       # 火箭排放指标

# =============================================================================
#                           计算函数
# =============================================================================

def ficon_curve(LDN):
    """FICON曲线: 高烦恼人口百分比"""
    return 100 / (1 + exp(11.13 - 0.141 * LDN))

def plan_C_stats(alpha):
    """计算混合方案的时间、成本期望和标准差"""
    # 电梯部分时间期望
    if alpha > 0:
        E_T_E = alpha * M / (E_E * (1 - p_E))
        Var_T_E = alpha * M * p_E / (E_E * (1 - p_E)**2)
    else:
        E_T_E = 0
        Var_T_E = 0
    
    # 火箭部分时间期望
    if alpha < 1:
        E_T_R = (1 - alpha) * M / P_avg / (N_rocket * q_R)
        Var_T_R = (1 - q_R) * ((1 - alpha) * M)**2 / (N_rocket**3 * q_R**3 * P_avg**2)
    else:
        E_T_R = 0
        Var_T_R = 0
    
    # 总时间 = max(E[T_E], E[T_R])
    E_T_C = max(E_T_E, E_T_R)
    Var_T_C = Var_T_E if E_T_E >= E_T_R else Var_T_R
    Std_T_C = sqrt(Var_T_C) if Var_T_C > 0 else 0
    
    # 电梯部分成本期望
    if alpha > 0:
        E_C_E = (alpha * M * C_E_cost + 
                 alpha * M * F_E / E_E + 
                 alpha * M * p_E * val_rep_E / (E_E * (1 - p_E))) / 10**8
        Var_C_E = (alpha**2 * M * p_E * val_rep_E**2 / (E_E * (1 - p_E)**2)) / 10**16
    else:
        E_C_E = 0
        Var_C_E = 0
    
    # 火箭部分成本期望
    if alpha < 1:
        M_R = (1 - alpha) * M
        E_C_R = (M_R * C_R_cost / q_R + 
                 val_R * M_R * (1 - q_R) / (P_avg * q_R) + 
                 M_R / (N_rocket * q_R * P_avg)) / 10**8
        Var_C_R = ((1 - alpha)**2 * 
                   (C_R_cost**2 * P_avg * M * (1 - q_R) / q_R**2 + 
                    val_R**2 * M * (1 - q_R) / (P_avg * q_R**2) + 
                    (1 - q_R) * M**2 / (N_rocket**3 * q_R**3 * P_avg**2))) / 10**16
    else:
        E_C_R = 0
        Var_C_R = 0
    
    # 总成本
    E_C_C = E_C_E + E_C_R
    Var_C_C = Var_C_E + Var_C_R
    Std_C_C = sqrt(Var_C_C) if Var_C_C > 0 else 0
    
    return E_T_C, Std_T_C, E_C_C, Std_C_C

def env_score(alpha):
    """计算混合方案的环境综合指标"""
    # 用地 (线性加权)
    A_mix = alpha * A_E + (1 - alpha) * A_R
    
    # 噪音 (声能量对数加权)
    if alpha > 0 and alpha < 1:
        LDN_mix = 10 * np.log10(alpha * 10**(LDN_E/10) + (1-alpha) * 10**(LDN_R/10))
    elif alpha == 1:
        LDN_mix = LDN_E
    else:
        LDN_mix = LDN_R
    HA_mix = ficon_curve(LDN_mix)
    
    # 排放 (线性加权)
    Pol_mix = alpha * Pol_E + (1 - alpha) * Pol_R
    
    return A_mix, HA_mix, Pol_mix

def normalize(arr):
    """Min-Max 归一化到 [0, 1]"""
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min < 1e-10:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)

# =============================================================================
#                           主程序
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("生成决策综合得分表格")
    print("=" * 60)
    
    # 定义网格
    alpha_values = np.linspace(0, 1, 101)  # 0%, 1%, 2%, ..., 100%
    w_E_values = np.linspace(0, 0.5, 51)   # 0%, 1%, 2%, ..., 50%
    
    print(f"α 范围: 0% ~ 100%, 共 {len(alpha_values)} 个值")
    print(f"w_E 范围: 0% ~ 50%, 共 {len(w_E_values)} 个值")
    
    # 预计算所有 alpha 对应的指标
    print("\n计算各 α 值对应的指标...")
    
    E_T_arr = np.zeros(len(alpha_values))
    Std_T_arr = np.zeros(len(alpha_values))
    E_C_arr = np.zeros(len(alpha_values))
    Std_C_arr = np.zeros(len(alpha_values))
    A_arr = np.zeros(len(alpha_values))
    HA_arr = np.zeros(len(alpha_values))
    Pol_arr = np.zeros(len(alpha_values))
    
    for i, alpha in enumerate(alpha_values):
        E_T, Std_T, E_C, Std_C = plan_C_stats(alpha)
        A_mix, HA_mix, Pol_mix = env_score(alpha)
        
        E_T_arr[i] = E_T
        Std_T_arr[i] = Std_T
        E_C_arr[i] = E_C
        Std_C_arr[i] = Std_C
        A_arr[i] = A_mix
        HA_arr[i] = HA_mix
        Pol_arr[i] = Pol_mix
    
    # 归一化
    E_C_norm = normalize(E_C_arr)
    E_T_norm = normalize(E_T_arr)
    Std_C_norm = normalize(Std_C_arr)
    Std_T_norm = normalize(Std_T_arr)
    A_norm = normalize(A_arr)
    HA_norm = normalize(HA_arr)
    Pol_norm = normalize(Pol_arr)
    
    # 环境综合指标 (w_emission=0.7, w_noise=0.2, w_land=0.1)
    Env_norm = 0.7 * Pol_norm + 0.2 * HA_norm + 0.1 * A_norm
    
    print("归一化完成")
    
    # 计算得分矩阵
    print("\n计算得分矩阵...")
    
    # 创建得分矩阵 (行: alpha, 列: w_E)
    score_matrix = np.zeros((len(alpha_values), len(w_E_values)))
    
    for j, w_E in enumerate(w_E_values):
        # 计算该 w_E 下的各项权重 (按 7:5:5:3 比例)
        remaining = 1 - w_E
        wc = 7 / 20 * remaining
        wt = 5 / 20 * remaining
        wsc = 5 / 20 * remaining
        wst = 3 / 20 * remaining
        
        # 计算所有 alpha 的得分
        scores = (wc * E_C_norm + 
                  wt * E_T_norm + 
                  wsc * Std_C_norm + 
                  wst * Std_T_norm + 
                  w_E * Env_norm)
        
        score_matrix[:, j] = scores
    
    # 创建 DataFrame
    # 列名: w_E 值 (百分比形式)
    col_names = [f"w_E={w*100:.0f}%" for w in w_E_values]
    # 行名: alpha 值 (百分比形式)
    row_names = [f"α={a*100:.0f}%" for a in alpha_values]
    
    df = pd.DataFrame(score_matrix, index=row_names, columns=col_names)
    
    # 保存为 CSV
    output_path = "../results/P4/score_table.csv"
    df.to_csv(output_path, encoding='utf-8-sig')
    
    print(f"\n表格已保存至: {output_path}")
    print(f"表格大小: {df.shape[0]} 行 × {df.shape[1]} 列")
    
    # 显示表格摘要
    print("\n" + "=" * 60)
    print("表格预览 (部分)")
    print("=" * 60)
    
    # 显示角落数据
    print("\n左上角 (α=0~10%, w_E=0~5%):")
    print(df.iloc[:11, :6].to_string())
    
    print("\n右下角 (α=90~100%, w_E=45~50%):")
    print(df.iloc[-11:, -6:].to_string())
    
    # 找到每列的最优 alpha
    print("\n" + "=" * 60)
    print("每个 w_E 对应的最优 α*")
    print("=" * 60)
    
    for j, w_E in enumerate(w_E_values):
        best_alpha_idx = np.argmin(score_matrix[:, j])
        best_alpha = alpha_values[best_alpha_idx]
        best_score = score_matrix[best_alpha_idx, j]
        if j % 5 == 0:  # 每5%打印一次
            print(f"w_E = {w_E*100:5.1f}%: α* = {best_alpha*100:6.2f}%, Score = {best_score:.4f}")
    
    print("\n完成！")
