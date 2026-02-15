# MCM/ICM 2026 Project TODO List

## Phase 0: Preparation (Hour 0-1)
- [x] **Initialize Project Structure** <!-- id: 0 -->
- [ ] **Import Problem Statement** (Prometheus) <!-- id: 1 -->
  - [ ] Copy full text to `doc/problem_description.md`
  - [ ] Extract key requirements and constraints

## Phase 1: Strategy & Research (Hour 1-4)
- [ ] **Deconstruct Problem** (Prometheus) <!-- id: 2 -->
  - [ ] Identify core conflict and variables
  - [ ] Propose 3 potential models (e.g., Differential Eq, CA, ML)
  - [ ] Define data requirements list
- [ ] **Literature Review** (Atlas) <!-- id: 3 -->
  - [ ] Search & Download 10-20 relevant papers to `references/`
    - *Tip*: Run `python download_papers.py` to get 2025 O-Prize papers.
  - [ ] Summary report: "Key formulas and parameters from O-award papers"
- [ ] **Data Acquisition** (Sisyphus) <!-- id: 4 -->
  - [ ] Search for datasets (Official/Open Source)
  - [ ] Write crawler/scraper scripts
  - [ ] Clean data using `data-cleaner` skill
  - [ ] Save processed data to `data/processed.csv`

## Phase 2: Modeling & Simulation (Hour 5-48)
- [ ] **Model 1: Baseline Model** (Sisyphus) <!-- id: 5 -->
  - [ ] Define mathematical formulation (LaTeX)
  - [ ] Implement in `code/model_baseline.py`
  - [ ] Run initial simulation/validation
- [ ] **Model 2: Advanced/Optimization Model** (Sisyphus) <!-- id: 6 -->
  - [ ] Select advanced algorithm (e.g., Genetic Algo, NSGA-II)
  - [ ] Implement in `code/model_advanced.py`
  - [ ] Compare with Baseline Model
- [ ] **Parameter Calibration** <!-- id: 7 -->
  - [ ] Use `automated-sweep` to fit parameters to history data
  - [ ] Use `bayesian-inversion` for uncertainty quantification (if applicable)

## Phase 3: Validation & Sensitivity (Hour 48-72)
- [ ] **Sensitivity Analysis** (Sisyphus) <!-- id: 8 -->
  - [ ] Run `sensitivity-master` (Sobol/Morris indices)
  - [ ] Generate Tornado Plot or Spider Plot
- [ ] **Robustness Check** <!-- id: 9 -->
  - [ ] Test extreme conditions (Stress testing)
  - [ ] Verify stability of equilibrium points
- [ ] **Uncertainty Quantification** <!-- id: 10 -->
  - [ ] Run `monte-carlo-engine` (95% Confidence Intervals)

## Phase 4: Visualization & Writing (Hour 72-90)
- [ ] **Visual Production** (Visual-Engineer) <!-- id: 11 -->
  - [ ] Generate "Publication Quality" charts (300dpi, Times New Roman)
  - [ ] Create Process Flowchart (Web-Artifacts/Draw.io)
- [ ] **Paper Drafting** (Sisyphus + Atlas) <!-- id: 12 -->
  - [ ] **Introduction**: Background, Restatement, Literature Review
  - [ ] **Methodology**: Assumptions, Notations, Model Derivation
  - [ ] **Results**: Tables, Figures, Error Analysis
  - [ ] **Discussion**: Strengths, Weaknesses, Future Work
- [ ] **Memo/Summary Sheet** (Prometheus) <!-- id: 13 -->
  - [ ] **CRITICAL**: Polish the 1-page summary (The most important part!)

## Phase 5: Final Polish & Submission (Hour 90-96)
- [ ] **Final Review** (Atlas) <!-- id: 14 -->
  - [ ] Logic consistency check (Code vs Text)
  - [ ] Citation check
  - [ ] Typos and grammar fix
- [ ] **Submission Assembly** <!-- id: 15 -->
  - [ ] Compile PDF (Main Paper)
  - [ ] Compile Code Appendix
  - [ ] Verify file requirements (Control #, Naming)
