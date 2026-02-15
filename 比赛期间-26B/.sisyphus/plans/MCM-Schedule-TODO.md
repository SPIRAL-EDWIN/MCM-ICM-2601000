# MCM/ICM 2026: Master Schedule & Action Plan (China Edition)

**Integrated 96-Hour Strategy**
- **Timezone**: CST (UTC+8)
- **Start**: Fri Jan 30, 09:00 (Team Start) | **End**: Tue Feb 3, 10:00 (Hard Stop)
- **Status**: `Active`

## 🚦 AI Execution Rules (MANDATORY)

**CRITICAL CONSTRAINTS for Sisyphus/Prometheus AI:**
1. ✅ **FEEDBACK GATE**: Every task marked with 🔴 requires **STOP and WAIT for user approval** before proceeding.
2. ✅ **NO AUTO-CHAIN**: Cannot complete multiple checklist items consecutively without user confirmation.
3. ✅ **DECISION POINTS**: All model/parameter choices must present options and wait for user selection.
4. ✅ **TODO TRANSPARENCY**: Use `todowrite` tool to track every task. Mark `in_progress` → `completed` only after user verifies.

**Violation = Task Rejection. User must explicitly say "proceed" or "continue" to advance.**

---

## 📅 Day 1: Selection & Foundation
**Friday, Jan 30**

### Phase 0: Intelligence Gathering (06:00 - 09:00)
*Status: Pre-Game*
*   **06:00**: Problems Released.
*   **06:00 - 09:00**: **[Team Sleep / Breakfast]** (Ensure full energy).

### Phase 1: The Decision (09:00 - 12:00)
*Goal: Lock one problem by noon.*
*   **09:00 - 10:00**: **Group Read & Debate**
    - [ ] **Import Problem** (Prometheus): Copy text to `doc/problem_description.md`.
    - [ ] **Filter**: Eliminate problems with missing data/obscure physics.
    - 🔴 **FEEDBACK GATE #1**: Present filtered problem list → Wait for user to select Top 2 candidates.
*   **10:00 - 11:00**: **Feasibility Check**
    - [ ] **Deconstruct** (Prometheus): Identify core conflicts and variables.
    - [ ] **Data Check** (Coder): Verify data availability for Top 2 choices.
    - 🔴 **FEEDBACK GATE #2**: Present feasibility analysis → Wait for user to lock final problem choice.
*   **11:00 - 12:00**: **Lock Problem**
    - [ ] **Decision**: Vote and Commit. Output: "Problem X Selected".

### Phase 2: The Skeleton (12:00 - 23:00)
*Goal: Data ready + Baseline Model conceptualized.*
*   **12:00 - 13:30**: **Lunch Break (Relax)**
*   **13:30 - 15:00**: **Deconstruction**
    - [ ] **Strategy**: Define input/output variables & sub-problems (3-4 tasks).
    - [ ] **Model Selection**: Propose 3 potential models (DiffEq, CA, ML, etc.).
    - 🔴 **FEEDBACK GATE #3**: Present 3 model candidates with pros/cons → Wait for user to select primary model + backup.
*   **15:00 - 18:00**: **Literature & Data Heist**
    - [ ] **Lit Review** (Atlas): Search/Download 10-20 papers to `references/`.
    - [ ] **Summary**: Extract key formulas/parameters from O-award papers.
    - [ ] **Data Acquisition** (Sisyphus): Write scrapers, clean data, save to `data/processed.csv`.
    - 🔴 **FEEDBACK GATE #4**: Present literature summary + data sources → Wait for user to confirm data strategy.
*   **18:00 - 19:00**: **Dinner & Sync**
*   **19:00 - 23:00**: **Baseline Model**
    - [ ] **Model 1 Formulation**: Define math (LaTeX) for the simplest version.
    - [ ] **Initial Stats**: Run basic data visualization.
    - 🔴 **FEEDBACK GATE #5**: Present baseline model formulation → Wait for user to approve before coding.

### 🛌 SLEEP BLOCK 1 (23:00 - 06:00)
*   **Constraint**: 7 Hours mandatory. Phones off.

---

## 📅 Day 2: The Engine Room
**Saturday, Jan 31**

### Phase 3: The Grind (06:00 - 18:00)
*Goal: Solve Task 1 & 2.*
*   **06:00 - 08:00**: **Sync**: Review baseline results.
*   **08:00 - 12:00**: **Solving Task 1 (Baseline)**
    - [ ] **Implement Model 1** (Sisyphus): Code in `code/model_baseline.py`.
    - [ ] **Validation**: Run initial simulation.
    - 🔴 **FEEDBACK GATE #6**: Present simulation results → Wait for user to verify correctness before writing.
    - [ ] **Writing**: Draft "Data Overview" & "Model 1 Formulation".
*   **12:00 - 13:30**: **Lunch Break (Nap encouraged)**
*   **13:30 - 18:00**: **Solving Task 2 (Advanced)**
    - [ ] **Select Algorithm**: Genetic Algo, NSGA-II, or advanced Stats.
    - 🔴 **FEEDBACK GATE #7**: Present algorithm options → Wait for user to select approach.
    - [ ] **Implement Model 2** (Sisyphus): Code in `code/model_advanced.py`.
    - [ ] **Comparison**: Generate metrics comparing Model 1 vs Model 2.
    - 🔴 **FEEDBACK GATE #8**: Present comparison results → Wait for user to approve before visualizing.

### Phase 4: Visualization First (18:00 - 23:00)
*Goal: 3 Publication-Quality Charts.*
*   **18:00 - 19:00**: **Dinner**
*   **19:00 - 23:00**: **Result Production**
    - [ ] **Visual Production** (Visual-Engineer): Generate 300dpi charts (Times New Roman).
    - 🔴 **FEEDBACK GATE #9**: Present draft charts → Wait for user to approve style/content before LaTeX integration.
    - [ ] **Integration**: Insert charts into `doc/main.tex`. Do not leave blank space.

### 🛌 SLEEP BLOCK 2 (23:00 - 06:00)
*   **Constraint**: 7 Hours mandatory.

---

## 📅 Day 3: Complexity & Sensitivity
**Sunday, Feb 1**

### Phase 5: Deep Dive (06:00 - 18:00)
*Goal: O-Prize Analysis & Abstract V1.*
*   **06:00 - 12:00**: **Solving Task 3 (Analysis)**
    - [ ] **Parameter Calibration**: Use `automated-sweep` to fit history data.
    - [ ] **Sensitivity Analysis**: Run `sensitivity-master` (Sobol/Morris indices).
    - [ ] **Robustness**: Stress test extreme conditions.
    - [ ] **Uncertainty**: Run `monte-carlo-engine` (95% CI).
    - 🔴 **FEEDBACK GATE #10**: Present all analysis results (calibration, sensitivity, uncertainty) → Wait for user to verify before writing abstract.
*   **12:00 - 13:30**: **Lunch Break (Brain reset)**
*   **13:30 - 18:00**: **The Abstract (First Pass)**
    - [ ] **CRITICAL**: Writer drafts Abstract V1.
    - 🔴 **FEEDBACK GATE #11**: Present Abstract V1 draft → Wait for user to approve structure/claims.
    - [ ] **Final Sims**: Coder finishes all simulation runs.

### Phase 6: Assembly (18:00 - 23:00)
*Goal: Full Draft Assembly.*
*   **18:00 - 19:00**: **Dinner**
*   **19:00 - 23:00**: **Drafting**
    - [ ] **Merge Sections**: Intro + Method + Results + Discussion.
    - [ ] **References**: Check citations against `references/` folder.
    - [ ] **Page Count**: Check if 20-25 pages reached.
    - 🔴 **FEEDBACK GATE #12**: Present full draft structure → Wait for user to review flow and identify gaps.

### 🛌 SLEEP BLOCK 3 (23:00 - 06:00)
*   **Constraint**: 7 Hours mandatory. Last sleep.

---

## 📅 Day 4: The Final Sprint (All-Nighter)
**Monday, Feb 2**

### Phase 7: Polish & Perfect (06:00 - 18:00)
*Goal: Perfect Abstract & Flow.*
*   **06:00 - 12:00**: **Review & Rewrite**
    - [ ] **Read Aloud**: Fix flow and awkward sentences.
    - [ ] **Captions**: Ensure every figure has a self-explanatory caption.
    - 🔴 **FEEDBACK GATE #13**: Present revised draft → Wait for user to approve language quality.
*   **12:00 - 13:30**: **Lunch Break (High Protein)**
*   **13:30 - 18:00**: **Abstract Final Polish**
    - [ ] **Memo/Summary Sheet**: Spend 2 hours refining. This page determines the prize.
    - [ ] **Checklist**: Method? Results? Metrics? Conclusion?
    - 🔴 **FEEDBACK GATE #14**: Present final Abstract/Summary → Wait for user's final approval before formatting phase.

### Phase 8: The Red Zone (18:00 - 06:00 Tue)
*Goal: Formatting & Safety.*
*   **18:00 - 20:00**: **Dinner & Power Nap** (Optional)
*   **20:00 - 24:00**: **Formatting**
    - [ ] **Style**: Check Margins, Font, Page Numbers.
    - [ ] **Control Sheet**: Verify Team Control Number.
    - 🔴 **FEEDBACK GATE #15**: Present formatted PDF → Wait for user to verify no content errors introduced.
*   **00:00 - 04:00**: **Zombie Zone (Caution)**
    - [ ] **NO NEW MATH**. Formatting only.
    - [ ] **Sanity Check**: Verify file names and MD5 hashes.

---

## 📅 Day 5: Submission
**Tuesday, Feb 3**

### Phase 9: Upload (06:00 - 10:00)
*   **06:00 - 07:00**: **Final PDF**
    - [ ] **Generate**: Compile Main Paper & Code Appendix.
    - [ ] **Verify**: Open on 3 different devices.
    - 🔴 **FEEDBACK GATE #16 (FINAL)**: Present submission package → Wait for user's GO signal before upload.
*   **07:00 - 08:00**: **Upload Attempt 1**
    - [ ] **Upload**: Submit to COMAP early (avoid server crash).
*   **08:00 - 09:00**: **Backup & Verification**
    - [ ] **Email/Forms**: Complete any secondary requirements.
*   **09:00**: **EDIT DEADLINE**.
*   **10:00**: **HARD STOP**.

---

## 🔍 Daily Checklist (Project Pulse)

- [ ] **Fri 23:00**: Model 1 math defined?
- [ ] **Sat 23:00**: 10 pages written?
- [ ] **Sun 23:00**: Abstract V1 written?
- [ ] **Mon 20:00**: Code frozen? (No more changes)
- [ ] **Tue 08:00**: Uploaded?

---

## 📋 FEEDBACK GATE Summary (Quick Reference)

| Gate # | Phase | What AI Presents | What User Decides |
|--------|-------|------------------|-------------------|
| 🔴 #1 | Problem Selection | Filtered problem list | Select Top 2 candidates |
| 🔴 #2 | Feasibility | Feasibility analysis | Lock final problem |
| 🔴 #3 | Model Selection | 3 model candidates + pros/cons | Select primary + backup model |
| 🔴 #4 | Literature/Data | Literature summary + data sources | Confirm data strategy |
| 🔴 #5 | Baseline Model | Model formulation (math) | Approve before coding |
| 🔴 #6 | Simulation | Baseline simulation results | Verify correctness |
| 🔴 #7 | Algorithm | Advanced algorithm options | Select approach |
| 🔴 #8 | Comparison | Model 1 vs Model 2 metrics | Approve before viz |
| 🔴 #9 | Visualization | Draft charts | Approve style/content |
| 🔴 #10 | Analysis | All analysis results | Verify before abstract |
| 🔴 #11 | Abstract V1 | Abstract draft | Approve structure/claims |
| 🔴 #12 | Draft Assembly | Full draft structure | Review flow, identify gaps |
| 🔴 #13 | Polish | Revised draft | Approve language quality |
| 🔴 #14 | Final Abstract | Final Abstract/Summary | Final approval |
| 🔴 #15 | Formatting | Formatted PDF | Verify no content errors |
| 🔴 #16 | Submission | Submission package | GO signal for upload |

**Total Feedback Gates: 16**
**Average per day: 4 gates**
**Enforcement: AI MUST stop at each gate and wait for explicit "proceed" command.**
