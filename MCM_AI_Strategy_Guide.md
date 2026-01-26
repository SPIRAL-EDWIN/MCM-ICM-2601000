# 美赛 (MCM/ICM) AI 战术手册 (OpenCode Edition)

> **核心理念**：不要把 AI 仅仅当作聊天机器人，要把它当作一位**不知疲倦、全能的第四位队友**（工程师/研究员）。
> **关键策略**：分工明确，流水线化，闭环工作。

---

## 一、 AI 战术分工 (Role Configuration)

在 `oh-my-opencode` 体系中，我们将任务拆解为三个角色，各司其职。请确保你的 `.opencode/config.json` 已正确配置。

| 模式 | 代号 | 角色定位 | 核心职责 | 推荐模型 |
| :--- | :--- | :--- | :--- | :--- |
| **Prometheus** | 普罗米修斯 | **大脑 / 规划者** | 题目拆解、逻辑推理、数学建模战略、模型优劣势分析 | **OpenAI o1** <br> (或 DeepSeek-R1) |
| **Sisyphus** | 西西弗斯 | **手 / 执行者** | 代码实现、数据清洗、运行脚本、LaTeX 论文写作、Git 管理 | **Gemini 3 Pro** <br> (Antigravity) |
| **Atlas** | 阿特拉斯 | **记忆 / 负重者** | 海量文献综述、数据集文档解析、跨文档逻辑检查 (RAG) | **Gemini 3 Pro** <br> (Antigravity) |

---

## 二、 赛程全流程实战指南 (Workflow)

### Phase 1: 破题与战略 (Hours 0-4)
**核心模式**: `Prometheus`

1.  **动作**: 将题目原文完整输入。
2.  **Prompt 示例**:
    > "作为数学建模专家，请分析这道题的核心矛盾。1. 识别变量和约束条件；2. 推荐3种可能的数学模型（如微分方程、元胞自动机、机器学习等）并比较优劣；3. 列出我们需要寻找的数据清单。"
3.  **产出**: 《解题战略书》 (Markdown)，明确数据需求和算法路径。

### Phase 2: 资料搜集与综述 (Hours 4-12)
**核心模式**: `Atlas` (文献) & `Sisyphus` (数据)

1.  **文献综述 (Atlas)**:
    *   **动作**: 将下载的 10-20 篇 PDF 放入 `references/` 文件夹。
    *   **Prompt**: "阅读 `references/` 下的所有论文，总结它们在解决类似问题时使用的核心公式，并提取出与本题相关的参数设定。"
2.  **数据爬取 (Sisyphus)**:
    *   **Prompt**: "使用 `web_search` 查找关于 [主题] 的开源数据集。编写 Python 脚本爬取数据并保存为 CSV。"

### Phase 3: 模型构建与求解 (Day 1-3)
**核心模式**: `Sisyphus`

*这是最耗时的部分。不要让 AI “写代码”，要让它 “开发项目”。*

1.  **项目初始化**:
    *   让 Sisyphus 建立结构：`/code` (代码), `/data` (数据), `/results` (结果), `/doc` (文档)。
2.  **核心实现**:
    *   **Prompt**: "基于 Prometheus 建议的 [算法名称]，在 `/code/model.py` 中实现模型。先写伪代码，确认后实现。"
    *   **Debug**: 报错直接丢回 Log，让 Sisyphus 自动修复。

### Phase 4: 学术级可视化 (持续进行)
**核心模式**: `Sisyphus` (配合 `visual-engineering` 技能)

*   **目标**: 产出 O 奖级别的图表。
*   **Prompt**: "使用 Seaborn 绘制预测结果。要求：学术风格，字体 Times New Roman，配色方案 'viridis'，分辨率 300dpi，保存到 `/results/fig1.png`。"

### Phase 5: 论文写作 (Day 3-4)
**核心模式**: `Sisyphus` (LaTeX 专家) & `Atlas` (逻辑检查)

1.  **Sisyphus 写作**:
    *   **动作**: 上传 MCM/ICM LaTeX 模板。
    *   **Prompt**: "根据 `/code/model.py` 的逻辑和 `/results/fig1.png` 的结果，在 `section_model.tex` 中编写'模型建立'章节。公式使用 LaTeX 格式。"
2.  **Atlas 检查**:
    *   **Prompt**: "检查 `paper.tex` 和 `code/` 的一致性，指出论文中描述的算法与实际代码是否有出入。"
3.  **Prometheus 润色**:
    *   **Prompt**: "作为美赛评委，对摘要 (Memo) 进行润色。要求：语言地道、逻辑强、突出创新点。"

---

## 三、 高级技巧 (Pro Tips)

### 1. 动态加载技能 (Delegate Task)
在 Sisyphus 模式下，使用 `delegate_task` 加载特定技能以提高专业度：

*   **数据处理**: `load_skills=["data-scientist"]`
    *   *Prompt*: "读取 data.csv，进行主成分分析(PCA)..."
*   **论文排版**: `load_skills=["latex-architect"]`
    *   *Prompt*: "调整表格格式，使其符合三线表规范..."
*   **作图**: `load_skills=["visual-artist"]`

### 2. 项目管理
*   **Todo List**: 强制要求 Sisyphus 维护 `TODO.md`，时刻清楚进度。
*   **Git 版本控制**: 让 Sisyphus 负责 `git commit`，防止多人协作覆盖代码。

### 3. 沟通原则
*   **把对话框当终端**：不要闲聊。每一条指令都应该指向文件操作或代码产出。
*   **遇到瓶颈**：不要死磕代码。切回 `Prometheus` (o1) 或咨询 `Oracle` 问思路，通了再写。

---

*祝浙大团队美赛顺利，拿下 O 奖！*
