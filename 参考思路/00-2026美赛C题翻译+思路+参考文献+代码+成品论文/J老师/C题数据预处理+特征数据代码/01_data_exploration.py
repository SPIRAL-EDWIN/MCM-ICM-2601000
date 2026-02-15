# -*- coding: utf-8 -*-
"""
2026年美赛C题 - 数据探索性分析（EDA）
《与星共舞》Dancing With the Stars 数据分析

作者：美赛建模团队
日期：2026-01-30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# ==================== 1. 数据读取 ====================
print("=" * 60)
print("1. 数据读取与基本信息")
print("=" * 60)

# 读取数据
data_path = r"C:\Users\jzh\Desktop\26美赛C题\2026_MCM-ICM_Problems\2026_MCM-ICM_Problems\2026_MCM_Problem_C_Data.csv"
df = pd.read_csv(data_path)

# 基本信息
print(f"\n数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"\n列名列表:")
for i, col in enumerate(df.columns):
    print(f"  {i+1:2d}. {col}")

print(f"\n数据类型:")
print(df.dtypes.value_counts())

# ==================== 2. 基础统计信息 ====================
print("\n" + "=" * 60)
print("2. 基础统计信息")
print("=" * 60)

# 赛季统计
print(f"\n赛季范围: 第 {df['season'].min()} 季 - 第 {df['season'].max()} 季")
print(f"总赛季数: {df['season'].nunique()} 个")
print(f"总参赛者数: {df.shape[0]} 人")

# 每赛季参赛者数量
season_counts = df['season'].value_counts().sort_index()
print(f"\n每赛季参赛者数量:")
print(season_counts.to_string())

# ==================== 3. 缺失值分析 ====================
print("\n" + "=" * 60)
print("3. 缺失值分析")
print("=" * 60)

# 统计每列的缺失值（包括N/A字符串）
missing_stats = []
for col in df.columns:
    # 检查真正的NaN
    nan_count = df[col].isna().sum()
    # 检查"N/A"字符串
    na_string_count = (df[col] == 'N/A').sum() if df[col].dtype == 'object' else 0
    # 检查0值（对于评分列，0可能表示已淘汰）
    zero_count = (df[col] == 0).sum() if df[col].dtype in ['int64', 'float64'] else 0
    
    total_missing = nan_count + na_string_count
    if total_missing > 0 or zero_count > 0:
        missing_stats.append({
            'column': col,
            'NaN': nan_count,
            'N/A_string': na_string_count,
            'zeros': zero_count,
            'total_missing': total_missing,
            'missing_pct': total_missing / len(df) * 100
        })

missing_df = pd.DataFrame(missing_stats)
if len(missing_df) > 0:
    print("\n缺失值统计（仅显示有缺失的列）:")
    print(missing_df.to_string(index=False))

# ==================== 4. 选手特征分析 ====================
print("\n" + "=" * 60)
print("4. 选手特征分析")
print("=" * 60)

# 4.1 行业分布
print("\n4.1 选手行业分布:")
industry_counts = df['celebrity_industry'].value_counts()
print(industry_counts.to_string())

# 4.2 来源国家/地区分布
print("\n4.2 选手来源国家/地区分布:")
country_counts = df['celebrity_homecountry/region'].value_counts()
print(country_counts.head(15).to_string())

# 4.3 年龄分布
print("\n4.3 选手年龄统计:")
age_stats = df['celebrity_age_during_season'].describe()
print(age_stats)

# 4.4 舞伴统计
print("\n4.4 专业舞伴出场次数 (Top 15):")
partner_counts = df['ballroom_partner'].value_counts()
print(partner_counts.head(15).to_string())

# ==================== 5. 比赛结果分析 ====================
print("\n" + "=" * 60)
print("5. 比赛结果分析")
print("=" * 60)

# 5.1 最终名次分布
print("\n5.1 最终名次分布:")
placement_counts = df['placement'].value_counts().sort_index()
print(placement_counts.to_string())

# 5.2 比赛结果类型
print("\n5.2 比赛结果类型统计:")
results_counts = df['results'].value_counts()
print(results_counts.to_string())

# ==================== 6. 评委评分分析 ====================
print("\n" + "=" * 60)
print("6. 评委评分分析")
print("=" * 60)

# 获取所有评分列
score_cols = [col for col in df.columns if 'judge' in col.lower() and 'score' in col.lower()]
print(f"\n评分列数量: {len(score_cols)}")
print(f"评分列格式: week{1-11}_judge{1-4}_score")

# 将N/A替换为NaN，0保留（表示已淘汰）
df_scores = df[score_cols].copy()
for col in score_cols:
    df_scores[col] = pd.to_numeric(df_scores[col].replace('N/A', np.nan), errors='coerce')

# 6.1 每位评委的评分统计
print("\n6.1 各评委评分统计:")
for judge_num in range(1, 5):
    judge_cols = [col for col in score_cols if f'judge{judge_num}_score' in col]
    judge_scores = df_scores[judge_cols].values.flatten()
    # 排除0和NaN
    valid_scores = judge_scores[(~np.isnan(judge_scores)) & (judge_scores > 0)]
    if len(valid_scores) > 0:
        print(f"  评委{judge_num}: 均值={valid_scores.mean():.2f}, 标准差={valid_scores.std():.2f}, "
              f"范围=[{valid_scores.min():.1f}, {valid_scores.max():.1f}], 有效样本={len(valid_scores)}")

# 6.2 各周评分变化趋势
print("\n6.2 各周平均评分趋势:")
weekly_avg = []
for week_num in range(1, 12):
    week_cols = [col for col in score_cols if f'week{week_num}_' in col]
    week_scores = df_scores[week_cols].values.flatten()
    valid_scores = week_scores[(~np.isnan(week_scores)) & (week_scores > 0)]
    if len(valid_scores) > 0:
        weekly_avg.append({
            'week': week_num,
            'mean': valid_scores.mean(),
            'std': valid_scores.std(),
            'count': len(valid_scores)
        })
        print(f"  第{week_num:2d}周: 均值={valid_scores.mean():.2f}, 有效样本={len(valid_scores)}")

# ==================== 7. 评委数量变化分析 ====================
print("\n" + "=" * 60)
print("7. 评委数量随赛季变化")
print("=" * 60)

# 分析每个赛季有几位评委
judge4_cols = [col for col in score_cols if 'judge4' in col]
for season in sorted(df['season'].unique()):
    season_df = df[df['season'] == season]
    season_scores = season_df[judge4_cols].copy()
    for col in judge4_cols:
        season_scores[col] = pd.to_numeric(season_scores[col].replace('N/A', np.nan), errors='coerce')
    
    # 检查是否有第4位评委的有效分数
    has_judge4 = season_scores.notna().any().any()
    valid_j4 = season_scores.notna().sum().sum()
    print(f"  第{season:2d}季: 第4评委{'有' if has_judge4 else '无'}评分 (有效分数数={valid_j4})")

# ==================== 8. 保存分析结果 ====================
print("\n" + "=" * 60)
print("8. 数据预处理")
print("=" * 60)

# 创建预处理后的数据框
df_clean = df.copy()

# 8.1 处理评分列：将N/A转为NaN，保留0（表示已淘汰）
for col in score_cols:
    df_clean[col] = pd.to_numeric(df_clean[col].replace('N/A', np.nan), errors='coerce')

# 8.2 计算每位选手每周的平均评委分
for week_num in range(1, 12):
    week_cols = [col for col in score_cols if f'week{week_num}_' in col]
    # 计算该周所有评委的平均分（忽略NaN）
    df_clean[f'week{week_num}_avg_score'] = df_clean[week_cols].mean(axis=1, skipna=True)

# 8.3 计算每位选手的总体平均分（排除0分周，即淘汰后的周）
avg_cols = [f'week{w}_avg_score' for w in range(1, 12)]
df_clean['overall_avg_score'] = df_clean[avg_cols].replace(0, np.nan).mean(axis=1, skipna=True)

# 8.4 计算每位选手参赛的周数
def count_active_weeks(row):
    count = 0
    for week_num in range(1, 12):
        avg = row[f'week{week_num}_avg_score']
        if pd.notna(avg) and avg > 0:
            count += 1
    return count

df_clean['active_weeks'] = df_clean.apply(count_active_weeks, axis=1)

print(f"\n预处理完成:")
print(f"  - 新增列: week1_avg_score ~ week11_avg_score (每周平均分)")
print(f"  - 新增列: overall_avg_score (总体平均分)")
print(f"  - 新增列: active_weeks (参赛周数)")

# 保存预处理后的数据
output_path = r"C:\Users\jzh\Desktop\26美赛C题\code\data_cleaned.csv"
df_clean.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n预处理数据已保存至: {output_path}")

# 打印预处理后的数据概览
print("\n预处理后数据概览:")
print(df_clean[['celebrity_name', 'season', 'placement', 'active_weeks', 'overall_avg_score']].head(20).to_string())

print("\n" + "=" * 60)
print("数据探索分析完成！")
print("=" * 60)
