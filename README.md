# MCM/ICM 2026 AI-Assisted Strategy & Workflow

> **Core Philosophy**: Don't treat AI as a chatbot. Treat it as a **tireless, versatile 4th teammate**.  
> **Repository Status**: `Active` | **Competition Phase**: `Preparation`

---

## 📂 Repository Structure

- **`/code`**: Source code for models, data processing, and analysis.
- **`/data`**: Raw and processed datasets.
- **`/results`**: Output files, figures, and simulation logs.
- **`/doc`**: Documentation, paper drafts, and notes.
- **`/references`**: Academic papers and PDF resources.
- **`.github/skills`**: **[MASTER SKILL BANK](.github/skills/README.md)** (Pre-written algorithms & tutorials).

---

## 🤖 AI Role Configuration (OhMyOpenCode)

We split tasks into three specialized roles. Ensure your mindset adapts to these personas.

| Codename | Role | Responsibility | Recommended Model |
| :--- | :--- | :--- | :--- |
| **Prometheus** | **Planner / Brain** | Problem deconstruction, logic reasoning, strategic modeling choices. | **OpenAI o1** / DeepSeek-R1 |
| **Sisyphus** | **Executor / Hand** | Coding, data cleaning, running scripts, LaTeX writing, Git ops. | **Gemini 3 Pro** (Antigravity) |
| **Atlas** | **Researcher / Memory** | Literature review, dataset parsing, logic consistency checks (RAG). | **Gemini 3 Pro** (Antigravity) |

---

## ⚡ The 96-Hour Workflow

### Phase 1: Strategy & Deconstruction (Hours 0-4)
**Role**: `Prometheus`
1.  **Input**: Full problem text.
2.  **Action**: Analyze core conflicts. Identify variables/constraints.
3.  **Output**: **Strategy Document** containing:
    *   3 Potential Models (e.g., DiffEq, Cellular Automata, ML).
    *   Data Requirements List.

### Phase 2: Research & Data (Hours 4-12)
**Role**: `Atlas` (Lit) & `Sisyphus` (Data)
1.  **Literature**: Search 10-20 papers. Place in `references/`. Summarize key formulas.
2.  **Data**: Crawl/Search datasets. Use `data-cleaner` skill to generate `processed.csv`.

### Phase 3: Modeling & Simulation (Day 1-3)
**Role**: `Sisyphus`
*This is the core engineering phase.*
1.  **Implementation**: Translate math to code (`code/model.py`).
2.  **Skill Injection**: Use **[.github/skills](.github/skills/README.md)** for instant algorithms:
    *   *Need weights?* → `entropy-weight-method`, `ahp-method`
    *   *Need prediction?* → `arima-forecaster`, `lstm-forecaster`
    *   *Need optimization?* → `genetic-algorithm`, `simulated-annealing`
    *   *Need inverse problem?* → `bayesian-inversion` (O-Prize Differentiator)

### Phase 4: Visualization (Continuous)
**Role**: `Sisyphus` + `visual-engineer`
*   **Goal**: Publication-quality charts (300 DPI, Times New Roman).
*   **Tools**: Matplotlib/Seaborn (Python), Web-Artifacts (D3/React).

### Phase 5: Writing & Polish (Day 3-4)
**Role**: `Sisyphus` (Writer) & `Atlas` (Reviewer)
1.  **Drafting**: Use `latex-coauthoring` skill.
2.  **Review**: Check consistency between Code and Text.
3.  **Memo**: Prometheus polishes the Summary Sheet (The most critical page).

---

## 🏆 O-Prize Protocols (High-Level)

For detailed implementation, see **[.github/skills/README.md](.github/skills/README.md)**.

### 1. Automated-Sweep (Parameter Scanning)
*   **Why**: Don't guess parameters. Fit them.
*   **Output**: Heatmaps of Loss Function vs Parameters.

### 2. Modular-Modeler (OOP Architecture)
*   **Why**: Avoid "Spaghetti Code".
*   **How**: `class Environment`, `class Agent`, `class Policy`.

### 3. Pareto-Frontier (Multi-Objective)
*   **Why**: Show trade-offs (e.g., Cost vs Environmental Impact).
*   **Output**: 2D scatter plot with non-dominated front.

### 4. Sensitivity-Master (Global Sensitivity)
*   **Why**: Prove robustness.
*   **Output**: Sobol Indices, Spider Plots.

### 5. Bayesian-Inversion (Uncertainty Quantification)
*   **Why**: **Key Differentiator**. Report confidence intervals, not just point estimates.
*   **Output**: "Parameter $\alpha = 5.2 \pm 0.3$ (95% CI)".

---

## 🛠️ Quick Command Reference

- **Clean Data**: "Invoke `data-cleaner` on `data/raw.csv`."
- **Search Code**: "Use `explore` to find similar patterns."
- **Check Skills**: "Read `.github/skills/README.md`."
- **Plan Tasks**: "Update `TODO.md`."

---
*Good luck, Zhejiang University Team! Let's get that O-Prize.*
