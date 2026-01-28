# MCM/ICM 2026 竞赛模型库完全指南 (Master Guide)

**版本**: 2.0 (2026年1月28日 - 融合版)  
**包含技能**: 28个核心技能  
**适用对象**: MCM/ICM 参赛队伍、数学建模爱好者

本文档整合了 **模型列表 (MODELS LIST)**、**速查卡 (QUICK REFERENCE)** 和 **优化集成指南 (OPTIMIZATION INTEGRATION)**，是比赛期间的唯一核心参考文档。

---

## 目录 (Table of Contents)

1. [⚡ 快速决策树 (Decision Tree)](#-快速决策树-decision-tree)
2. [⏱️ 时间预算表 (Time Budgets)](#-时间预算表-time-budgets)
3. [📅 第一阶段：前期数据处理](#-第一阶段前期数据处理-data-pre-processing)
4. [⚙️ 第二阶段：中期模型构建](#-第二阶段中期模型构建-mid-stage-modeling)
   - [A. 评价与决策](#a-评价与决策类-evaluation--decision)
   - [B. 预测与时序](#b-预测与时间序列类-forecasting--time-series)
   - [C. 机理与微分方程](#c-机理与微分方程类-mechanistic--differential-equations)
   - [D. 网络与图论](#d-网络与图论类-network--graph-theory)
   - [E. 优化与规划 (含深度集成指南)](#e-优化与规划类-optimization--planning)
5. [✅ 第三阶段：验证与分析](#-第三阶段后期验证与分析-validation--analysis)
6. [📝 第四阶段：论文产出](#-第四阶段论文产出-post-processing)
7. [🚀 经典调用流程](#-经典调用流程示例)
8. [🏆 O奖检查清单](#-o奖检查清单-o-prize-checklist)

---

## ⚡ 快速决策树 (Decision Tree)

```text
开始: 你需要做什么？

├─ "清洗数据" (Clean data)
│  ├─ 基础清洗 (缺失/异常) → data-cleaner
│  └─ 降维 (指标太多) → pca-analyzer
│
├─ "预测未来" (Forecast)
│  ├─ 大样本 (>50点) → arima-forecaster
│  ├─ 小样本 (4-10点) → grey-forecaster
│  └─ 复杂非线性 (>500点) → lstm-forecaster
│
├─ "做决策 / 排名" (Decision/Rank)
│  ├─ 客观权重 (数据驱动) → entropy-weight-method
│  ├─ 主观权重 (专家经验) → ahp-method
│  └─ 最终排名 → topsis-scorer
│
├─ "机器学习预测" (Predict with ML)
│  └─ 回归/特征重要性 → ml-regressor
│
├─ "机理建模" (Dynamics)
│  ├─ 单一种群增长 → logistic-growth
│  ├─ 双种群互动 (捕食/竞争) → lotka-volterra
│  ├─ 空间传播 (传染病/污染) → reaction-diffusion
│  └─ 通用方程求解 → differential-equations
│
├─ "网络与路径" (Network)
│  ├─ 找最短路径/路由 → shortest-path (Dijkstra/Floyd)
│  └─ 找关键节点/影响力 → network-centrality
│
├─ "优化求解" (Optimize)
│  ├─ 离散/组合问题 (TSP/排班) → genetic-algorithm
│  ├─ 连续多峰 (易陷局部最优) → simulated-annealing
│  ├─ 连续光滑 (追求速度) → particle-swarm
│  ├─ 多目标 (既要又要) → multi-objective-optimization
│  └─ 简单参数搜索 → automated-sweep
│
├─ "验证模型" (Validate)
│  ├─ 寻找最佳参数 → automated-sweep
│  ├─ 哪个参数最重要？→ sensitivity-master
│  ├─ 模型稳不稳定？→ robustness-check
│  ├─ 结果的不确定性范围？→ monte-carlo-engine
│  └─ 多目标权衡分析 → pareto-frontier
│
├─ "构建复杂系统" (Build)
│  └─ → modular-modeler
│
├─ "画图" (Figures)
│  └─ → visual-engineer
│
└─ "写论文" (Write)
   └─ → latex-transformer
```

---

## ⏱️ 时间预算表 (Time Budgets)

| 技能 | 最快耗时 | 典型耗时 | 包含调试 |
|------|---------|---------|---------|
| **基础工具** | | | |
| data-cleaner | 10 min | 20 min | 30 min |
| visual-engineer | 5 min | 10 min | 20 min |
| latex-transformer | 5 min | 10 min | 15 min |
| **评价类** | | | |
| entropy/topsis | 10 min | 15 min | 25 min |
| ahp-method | 15 min | 25 min | 40 min |
| pca-analyzer | 15 min | 20 min | 30 min |
| **预测类** | | | |
| grey-forecaster | 10 min | 15 min | 25 min |
| arima-forecaster | 20 min | 35 min | 60 min |
| lstm-forecaster | 45 min | 90 min | 2 hours |
| **机理类** | | | |
| logistic/lotka | 10-15 min | 25 min | 40 min |
| reaction-diffusion | 30 min | 50 min | 90 min |
| differential-eqn | 20 min | 35 min | 60 min |
| **网络类** | | | |
| shortest-path | 15 min | 25 min | 40 min |
| network-centrality | 20 min | 30 min | 50 min |
| **优化类** | | | |
| automated-sweep | 30 min | 1 hour | 2 hours |
| particle-swarm | 15 min | 25 min | 45 min |
| simulated-annealing | 20 min | 35 min | 60 min |
| genetic-algorithm | 30 min | 50 min | 90 min |
| multi-objective | 1 hour | 1.5 hours | 3 hours |
| **验证类** | | | |
| robustness-check | 30 min | 45 min | 1.5 hours |
| monte-carlo | 30 min | 1 hour | 2 hours |
| sensitivity | 1 hour | 2 hours | 4 hours |

---

## 📅 第一阶段：前期数据处理 (Data Pre-processing)

### 1. **data-cleaner** (数据清洗专家)
- **技能定位**: 自动化数据预处理工具，处理脏数据、缺失值和异常值。
- **核心要求**: 必须处理缺失值（插值/删除）和异常值（3-sigma/IQR）。
- **适用数据**: 原始的、未清洗的CSV/Excel文件。
- **使用场景**: 刚下载原始数据，需要生成 `processed.csv` 作为后续输入。
- **实现策略**: Pandas 向量化操作，自动生成清洗报告。

### 2. **pca-analyzer** (主成分分析器)
- **技能定位**: 降维工具，消除多重共线性。
- **核心要求**: 保留85%以上累计方差，并解释主成分含义。
- **适用数据**: 指标过多（>10个）且存在相关性的高维数据。
- **使用场景**: 评价指标太多，担心模型过拟合或难以解释。
- **实现策略**: sklearn PCA，输出特征值和载荷矩阵。

---

## ⚙️ 第二阶段：中期模型构建 (Mid-stage Modeling)

### A. 评价与决策类 (Evaluation & Decision)

#### 3. **entropy-weight-method** (熵权法)
- **定位**: 客观赋权，基于数据变异程度。
- **场景**: 不希望主观因素干扰权重时。常与 TOPSIS 连用。

#### 4. **ahp-method** (层次分析法)
- **定位**: 主观赋权，基于专家两两比较。
- **要求**: 一致性比例 CR < 0.1。
- **场景**: 处理难以量化的定性指标（如"政策力度"）。

#### 5. **topsis-scorer** (优劣解距离法)
- **定位**: 综合排名工具。
- **经典组合**: Entropy + AHP + TOPSIS (主客观点综合赋权)。
- **输出**: 各方案的相对贴近度得分及排名。

### B. 预测与时间序列类 (Forecasting & Time Series)

#### 6. **grey-forecaster** (灰色预测 GM(1,1))
- **定位**: 贫信息预测，小样本神器。
- **要求**: 样本量 4-10 个，数据全为正。
- **场景**: 历史数据极少（如仅有5年），需预测未来趋势。

#### 7. **arima-forecaster** (ARIMA 时序)
- **定位**: 经典统计预测，捕捉趋势和季节性。
- **要求**: 样本量 > 50，通过平稳性检验。
- **场景**: 数据量充足的经济、气象指标预测。

#### 8. **lstm-forecaster** (深度学习 LSTM)
- **定位**: 处理复杂非线性长序列。
- **要求**: 数据量 > 500，需划分训练/测试集。
- **场景**: 传统模型失效，需要高精度拟合复杂模式。

#### 9. **ml-regressor** (机器学习回归)
- **定位**: 随机森林/XGBoost 回归。
- **亮点**: 输出特征重要性 (Feature Importance)，增加可解释性。
- **场景**: 预测房价、销量，并找出关键驱动因素。

### C. 机理与微分方程类 (Mechanistic & Differential Equations)

#### 10. **logistic-growth** (Logistic 增长)
- **定位**: 资源受限的单一种群增长（S型）。
- **要求**: 估计环境承载力 K 值。
- **场景**: 人口、传染病早期、产品扩散。

#### 11. **lotka-volterra** (捕食者-猎物)
- **定位**: 双种群竞争或捕食关系。
- **要求**: 绘制相图 (Phase Portrait) 分析稳定性。
- **场景**: 生态平衡、商业竞争 (A公司 vs B公司)。

#### 12. **reaction-diffusion** (反应扩散)
- **定位**: 时空演化模型（时间+空间）。
- **输出**: 2D/3D 动态演化热力图。
- **场景**: 传染病空间传播、污染物扩散、生态入侵。

#### 13. **differential-equations** (通用求解器)
- **定位**: 统一的 ODE/PDE 数值解框架。
- **场景**: 标准模型无法满足，需自定义方程组时。

### D. 网络与图论类 (Network & Graph Theory)

#### 14. **shortest-path** (最短路径)
- **定位**: Dijkstra / Floyd / Bellman-Ford / A* 算法库。
- **场景**: 物流配送、应急救援、路由规划。
- **流程**: 构建邻接矩阵 → 算距离 → 输入优化模型。

#### 15. **network-centrality** (中心性分析)
- **定位**: 识别网络中的关键节点 (Key Nodes)。
- **指标**: 度 (Degree)、介数 (Betweenness)、接近 (Closeness)、特征向量 (Eigenvector)。
- **场景**: 找社交网络意见领袖、电网脆弱节点、超级传播者。

### E. 优化与规划类 (Optimization & Planning)

#### 16. **genetic-algorithm** (遗传算法)
- **特点**: 全局搜索，适合离散编码。
- **场景**: 选址、排班、路径规划。

#### 17. **simulated-annealing** (模拟退火)
- **特点**: 原理简单，擅长跳出局部最优。
- **场景**: 复杂函数寻优。

#### 18. **particle-swarm** (粒子群)
- **特点**: 收敛快，实现简单。
- **场景**: 连续参数拟合。

#### 19. **multi-objective-optimization** (多目标 NSGA-II)
- **定位**: 寻找 Pareto 前沿面。
- **场景**: "既要成本低，又要质量高" 的冲突目标优化。

---

#### 🔧 优化算法深度集成指南 (Deep Dive)

> **如何选择算法?**
> - **离散/组合问题** (TSP, 0/1规划) → **genetic-algorithm**
> - **多峰连续问题** (易陷局部最优) → **simulated-annealing**
> - **光滑连续问题** (追求速度) → **particle-swarm**
>
> **多目标算法原理**:
> - **NSGA-II** = 遗传算法 (GA) + Pareto 排序
>   - 使用 GA 的交叉变异 + 非支配排序 + 拥挤度距离
> - **MOEA/D** = 分解法 + 模拟退火 (SA) 接受准则
> - **MOPSO** = 粒子群 (PSO) + Pareto 档案集
>
> **推荐使用模式**:
> 1. 先用单目标算法 (GA/PSO) 跑通流程。
> 2. 如果发现有权衡 (Trade-off)，升级为多目标 (NSGA-II)。
> 3. 最后使用 `topsis-scorer` 从 Pareto 前沿中选出最优解。

---

## ✅ 第三阶段：后期验证与分析 (Validation & Analysis)

### 20. **sensitivity-master** (灵敏度分析)
- **定位**: Sobol / Morris 全局分析。
- **目的**: 识别关键参数，证明模型对参数扰动不敏感（鲁棒性）。
- **O奖标准**: 必须做。不仅仅是单因素分析。

### 21. **robustness-check** (鲁棒性检验)
- **定位**: 极端条件测试。
- **输出**: 龙卷风图 (Tornado Diagram)。
- **场景**: "如果我的假设稍微不成立，模型会崩溃吗？"

### 22. **monte-carlo-engine** (蒙特卡洛模拟)
- **定位**: 不确定性量化。
- **输出**: 95% 置信区间。
- **场景**: 输入参数带有随机性时。

### 23. **automated-sweep** (参数扫描)
- **定位**: 暴力网格搜索最优参数。
- **场景**: 模型参数未知，反向拟合历史数据。

### 24. **pareto-frontier** (Pareto 可视化)
- **定位**: 多目标权衡图。
- **场景**: 直观展示目标间的制约关系。

### 25. **modular-modeler** (模块化架构)
- **定位**: 复杂系统 OOP 框架。
- **场景**: 代码量大，需多人协作开发时使用。

---

## 📝 第四阶段：论文产出 (Post-processing)

### 26. **visual-engineer** (可视化专家)
- **定位**: 生成出版级 (300 DPI, Times New Roman) 图表。
- **场景**: 替换 Python 默认丑陋图表。

### 27. **latex-transformer** (LaTeX 转换)
- **定位**: Markdown 转 LaTeX 代码。
- **场景**: 快速生成公式、表格、引用。

---

## 🚀 经典调用流程示例

**场景 1：传染病防控 (Epidemic Control)**
1. **Data**: `data-cleaner` 清洗数据。
2. **Model**: `differential-equations` 建立 SIR 模型。
3. **Param**: `automated-sweep` 拟合感染率。
4. **Validation**: `sensitivity-master` 分析 R0 敏感性。
5. **Output**: `visual-engineer` 绘制预测曲线。

**场景 2：物流选址 (Logistics)**
1. **Network**: `data-cleaner` + `shortest-path` 构建距离矩阵。
2. **Analysis**: `network-centrality` 找候选中心。
3. **Optimize**: `genetic-algorithm` 求解最小成本选址。
4. **Eval**: `topsis-scorer` 综合评估方案。
5. **Output**: 选址地图。

**场景 3：政策制定 (Policy Making)**
1. **Index**: `pca-analyzer` 降维指标。
2. **Forecast**: `lstm-forecaster` 预测趋势。
3. **Optimize**: `multi-objective-optimization` 寻找经济/环境平衡点。
4. **Decision**: `pareto-frontier` + `robustness-check` 验证政策。

---

## 🏆 O奖检查清单 (O-Prize Checklist)

- [ ] **模型选择有理有据**: 使用决策树证明为什么选这个模型。
- [ ] **不确定性量化**: 必须有 Monte Carlo 或置信区间。
- [ ] **灵敏度分析**: 必须有 Sensitivity Analysis (Sobol/Robustness)。
- [ ] **图表质量**: 必须是出版级 (visual-engineer)。
- [ ] **代码规范**: 清晰、模块化。

---

*(本文档由 Sisyphus 自动生成，融合了 QUICK_REFERENCE, MODELS_LIST, OPTIMIZATION_INTEGRATION)*
