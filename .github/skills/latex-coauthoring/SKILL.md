---
name: latex-coauthoring
description: "MCM/ICM LaTeX Paper Co-Authoring Workflow. Use when the team needs to draft, edit, or polish sections of the competition solution paper in LaTeX. Specializes in: (1) Structuring the paper skeleton (sections, subsections, symbols table), (2) Drafting mathematical models with proper equation formatting, (3) Polishing the critical Summary Sheet (one-page memo), (4) Academic tone and LaTeX best practices. Assumes LaTeX/Overleaf workflow for MCM/ICM competitions."
---

# MCM/ICM LaTeX Co-Authoring Specialist

## Overview

You are the LaTeX Writing Expert for an MCM/ICM team. Your mission: help the team produce a **publication-quality solution paper** in LaTeX within 96 hours, with special emphasis on the **Summary Sheet** — the single most important page that determines whether judges read the full paper.

**Critical Understanding**: In MCM/ICM, the Summary Sheet is read first. If it fails to impress in 60 seconds, the paper is effectively disqualified from top prizes. The Summary Sheet must be polished **first** (outline) and **last** (final polish).

## Competition Context

**MCM/ICM Paper Requirements**:
- **Summary Sheet**: 1-page memo format, standalone, non-technical language
- **Main Paper**: Up to 25 pages including figures, excluding appendices
- **Structure**: Problem Restatement → Assumptions → Model Development → Results → Sensitivity Analysis → Conclusions
- **Tooling**: LaTeX (Overleaf or local compiler), must generate PDF

**Time Allocation** (typical):
- Hours 0-24: Problem analysis, data exploration (minimal writing)
- Hours 24-60: Model building, coding (draft structure in parallel)
- Hours 60-84: Results generation, paper drafting (heavy writing)
- Hours 84-96: Summary Sheet polish, final proofreading (critical phase)

## Core Workflow: Three-Stage Writing Process

### Stage 1: Structural Blueprint (Hours 24-36)

**Goal**: Define the paper skeleton before writing prose. This prevents scope creep and ensures logical flow.

#### Step 1.1: Problem Restatement
**Input**: User provides the original problem statement.
**Output**: A concise, clear restatement in 2-3 paragraphs.

**LaTeX Template**:
```latex
\section{Problem Restatement}

The problem requires us to [concise description of the core task]. Specifically, we must [list 2-3 key objectives]:

\begin{itemize}
    \item Objective 1: [description]
    \item Objective 2: [description]
    \item Objective 3: [description]
\end{itemize}

The solution must [key constraints or deliverables].
```

#### Step 1.2: Assumptions
**Goal**: List 3-7 core assumptions with justifications.

**LaTeX Template**:
```latex
\section{Assumptions}

To simplify the problem and ensure tractability, we make the following assumptions:

\begin{enumerate}
    \item \textbf{Assumption 1}: [Statement]. \\
    \textit{Justification}: [Why this is reasonable].
    
    \item \textbf{Assumption 2}: [Statement]. \\
    \textit{Justification}: [Why this is reasonable].
    
    % Add more as needed
\end{enumerate}
```

**Best Practices**:
- Avoid trivial assumptions ("We assume data is accurate")
- Justify non-obvious assumptions with literature or logic
- Flag simplifying assumptions that might affect results

#### Step 1.3: Notation Table
**Goal**: Create a comprehensive symbols table for all variables.

**LaTeX Template**:
```latex
\section{Notation}

\begin{table}[H]
\centering
\caption{Notation and Symbols}
\begin{tabular}{cl}
\toprule
\textbf{Symbol} & \textbf{Definition} \\
\midrule
$N$ & Total population size \\
$S(t)$ & Number of susceptible individuals at time $t$ \\
$I(t)$ & Number of infected individuals at time $t$ \\
$\beta$ & Transmission rate (infections per contact per day) \\
$\gamma$ & Recovery rate (1/day) \\
\bottomrule
\end{tabular}
\end{table}
```

**Required Packages**:
```latex
\usepackage{booktabs}  % For professional tables
\usepackage{float}     % For [H] placement
```

#### Step 1.4: Model Sections Outline
**Goal**: Define 2-3 distinct models or major components.

**LaTeX Template**:
```latex
\section{Model Development}

\subsection{Model 1: [Name, e.g., Predictive Model]}
\textit{Purpose}: [What this model does]

% Placeholder for equations and explanations

\subsection{Model 2: [Name, e.g., Optimization Model]}
\textit{Purpose}: [What this model does]

% Placeholder for equations and explanations

\subsection{Model Integration}
\textit{How models connect}: [Brief description]
```

---

### Stage 2: Section Drafting (Hours 60-80)

**Goal**: Convert raw math and logic into polished LaTeX sections.

#### Step 2.1: Equation Formatting

**Single Equations** (numbered, for reference):
```latex
The SIR model is governed by:
\begin{equation}
    \frac{dS}{dt} = -\beta S I
    \label{eq:sir_s}
\end{equation}

From Equation \eqref{eq:sir_s}, we observe that...
```

**Multi-line Equations** (aligned):
```latex
\begin{align}
    \frac{dS}{dt} &= -\beta S I \label{eq:sir_s} \\
    \frac{dI}{dt} &= \beta S I - \gamma I \label{eq:sir_i} \\
    \frac{dR}{dt} &= \gamma I \label{eq:sir_r}
\end{align}
```

**Inline Math** (for text flow):
```latex
The basic reproduction number $R_0 = \beta / \gamma$ determines epidemic spread.
```

**Piecewise Functions**:
```latex
\begin{equation}
    f(x) = 
    \begin{cases}
        0 & \text{if } x < 0 \\
        x^2 & \text{if } 0 \leq x < 1 \\
        1 & \text{if } x \geq 1
    \end{cases}
\end{equation}
```

#### Step 2.2: Algorithm Descriptions

**Use `algorithm2e` package**:
```latex
\usepackage[ruled,vlined]{algorithm2e}

\begin{algorithm}[H]
\caption{Genetic Algorithm for Optimization}
\KwIn{Population size $N$, generations $G$, mutation rate $p_m$}
\KwOut{Optimal solution $x^*$}

Initialize population $P_0$ randomly\;
\For{$g = 1$ \KwTo $G$}{
    Evaluate fitness for each individual in $P_g$\;
    Select parents using tournament selection\;
    Apply crossover and mutation\;
    Create new generation $P_{g+1}$\;
}
\Return best individual from $P_G$\;
\end{algorithm}
```

#### Step 2.3: Figure Integration

**Best Practices**:
- Save figures as `.png` (300 DPI) or `.pdf` (vector graphics)
- Use descriptive filenames: `sir_model_results.png`
- Always include captions that explain what the figure shows

**LaTeX Template**:
```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/sir_model_results.png}
    \caption{SIR Model Predictions: The model accurately captures the epidemic peak at day 45, with a maximum of 3,200 infected individuals. The shaded region represents 95\% confidence intervals from Monte Carlo simulations.}
    \label{fig:sir_results}
\end{figure}

As shown in Figure \ref{fig:sir_results}, the epidemic peaks around day 45.
```

#### Step 2.4: Academic Tone Guidelines

**Passive Voice** (preferred in formal writing):
- ✅ "The model was validated using historical data."
- ❌ "We validated the model using historical data."

**Objective Language**:
- ✅ "The results indicate a strong correlation ($r = 0.92$)."
- ❌ "Our amazing results show a super strong correlation."

**Precise Quantification**:
- ✅ "The algorithm converged in 47 iterations with a tolerance of $10^{-6}$."
- ❌ "The algorithm converged pretty quickly."

**Logical Connectors**:
- Use: "Furthermore", "Consequently", "In contrast", "Specifically"
- Avoid: "Also", "But", "So", "Basically"

---

### Stage 3: The Summary Sheet (Hours 84-92)

**Critical Phase**: This is the **last** thing written but the **first** thing read by judges.

#### Step 3.1: Summary Sheet Structure

**Format**: One-page memo (no section numbering, no references to paper sections).

**Template**:
```latex
\documentclass[12pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath}
\pagestyle{empty}

\begin{document}

\begin{center}
\Large \textbf{Summary Sheet}
\end{center}

\section*{Problem Overview}
[2-3 sentences: What is the problem? Why is it important?]

\section*{Our Approach}
[3-4 sentences: High-level description of models used, without technical jargon]

\section*{Key Results}
[Bullet points with specific numeric results]
\begin{itemize}
    \item Result 1: [e.g., "Our model predicts a 23\% reduction in costs"]
    \item Result 2: [e.g., "Sensitivity analysis shows the solution is robust to ±15\% parameter variations"]
    \item Result 3: [e.g., "The optimal strategy is X, which outperforms baseline by 34\%"]
\end{itemize}

\section*{Strengths and Limitations}
\textbf{Strengths}: [1-2 sentences] \\
\textbf{Limitations}: [1-2 sentences, honest about assumptions]

\section*{Conclusions}
[2-3 sentences: Practical implications and recommendations]

\end{document}
```

#### Step 3.2: Judge Perspective Review

**Critical Questions** (ask these before finalizing):
1. **Can a non-technical judge understand the first paragraph?**
   - If no: Simplify language, remove jargon.
2. **Are the numeric results prominent and specific?**
   - If no: Add percentages, concrete numbers, comparisons.
3. **Does it fit on one page with readable font (12pt)?**
   - If no: Cut unnecessary details, tighten prose.
4. **Does it convey confidence without arrogance?**
   - If no: Use phrases like "Our analysis suggests" instead of "We definitively prove".

#### Step 3.3: Final Polish Checklist

- [ ] No typos or grammatical errors
- [ ] No undefined acronyms (spell out on first use)
- [ ] No references to "Section 3.2" or "Figure 4" (Summary Sheet is standalone)
- [ ] Results are quantified with units
- [ ] Limitations are acknowledged (shows maturity)
- [ ] Printed version looks professional (test print to PDF)

---

## LaTeX Best Practices for MCM/ICM

### Document Class and Packages

**Recommended Preamble**:
```latex
\documentclass[12pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{graphicx}
\usepackage{float}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{cite}
\usepackage[ruled,vlined]{algorithm2e}

\title{Solution to MCM Problem [X]: [Title]}
\author{Team \# [Your Team Number]}
\date{\today}
```

### Float Placement

**Problem**: Figures/tables float to unexpected locations.
**Solution**: Use `[H]` (requires `\usepackage{float}`):
```latex
\begin{figure}[H]  % Forces "Here" placement
    \centering
    \includegraphics[width=0.7\textwidth]{figure.png}
    \caption{Caption text}
\end{figure}
```

### Citations

**Use BibTeX** for automatic formatting:
```latex
% In main.tex
\bibliographystyle{plain}
\bibliography{references}

% In references.bib
@article{smith2020,
    author = {Smith, John},
    title = {A Study on Epidemic Models},
    journal = {Journal of Mathematical Biology},
    year = {2020},
    volume = {45},
    pages = {123--145}
}

% In text
According to Smith \cite{smith2020}, the SIR model is effective for...
```

### Common LaTeX Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Undefined control sequence` | Typo in command name | Check spelling: `\frac{}{}` not `\frac{}` |
| `Missing $ inserted` | Math mode outside `$...$` | Wrap math: `$x^2$` not `x^2` |
| `Float too large` | Figure exceeds page height | Reduce `width=` parameter |
| `Citation undefined` | BibTeX not run | Compile sequence: `pdflatex → bibtex → pdflatex × 2` |

---

## Workflow Integration

### When to Use This Skill

**Trigger Phrases**:
- "Draft the Introduction section"
- "Format these equations in LaTeX"
- "Polish the Summary Sheet"
- "Create a notation table"
- "Convert this algorithm to LaTeX pseudocode"

### Handoff to Other Skills

- **After latex-coauthoring**: Use `visual-engineer` to create high-quality figures for `\includegraphics`
- **Before latex-coauthoring**: Use `xlsx` to generate data tables, `topsis-scorer` to get ranking results
- **Parallel with latex-coauthoring**: Use `pdf` to extract equations from literature for citation

---

## Time-Saving Tips for Competition

### Quick Drafting (Hours 60-72)
**Goal**: Get ideas on paper fast, polish later.

**Strategy**:
- Use `\section{}` and `\subsection{}` liberally to organize
- Write equations first, prose second
- Leave `[TODO: explain X]` markers for later
- Don't worry about perfect wording yet

### Rapid Equation Entry
**Use LaTeX shortcuts**:
- `\newcommand{\dd}[2]{\frac{d#1}{d#2}}` → `\dd{S}{t}` instead of `\frac{dS}{dt}`
- `\newcommand{\R}{\mathbb{R}}` → `\R^n` instead of `\mathbb{R}^n`

### Collaborative Editing (Overleaf)
**Best Practices**:
- **Assign sections**: Person A writes Model 1, Person B writes Model 2
- **Use comments**: `% TODO: Add sensitivity analysis here`
- **Track history**: Overleaf auto-saves, use "History" to revert mistakes
- **Avoid merge conflicts**: Don't edit the same paragraph simultaneously

---

## Output Standards

### File Organization
```
project/
├── main.tex              # Main paper
├── summary_sheet.tex     # Separate Summary Sheet file
├── references.bib        # BibTeX bibliography
├── figures/
│   ├── model_diagram.png
│   ├── results_plot.png
│   └── sensitivity_analysis.png
└── compiled/
    ├── main.pdf
    └── summary_sheet.pdf
```

### Quality Checklist (Before Submission)

**Content**:
- [ ] Problem Restatement is clear and concise
- [ ] All assumptions are justified
- [ ] All variables are defined in Notation table
- [ ] Equations are numbered and referenced in text
- [ ] Figures have descriptive captions
- [ ] Results are quantified with units and uncertainties
- [ ] Sensitivity analysis is included
- [ ] Summary Sheet is polished and standalone

**Formatting**:
- [ ] 12pt font, 1-inch margins
- [ ] All figures are high-resolution (300 DPI)
- [ ] Tables use `booktabs` style (professional)
- [ ] No overfull hbox warnings (text overflowing margins)
- [ ] Page count ≤ 25 pages (excluding appendices)

**LaTeX Compilation**:
- [ ] Compiles without errors
- [ ] All references resolved (no `[?]` in PDF)
- [ ] Hyperlinks work (if using `hyperref`)

---

## Advanced Techniques

### Multi-Column Layouts (for Summary Sheet)
```latex
\usepackage{multicol}

\begin{multicols}{2}
[Content flows across two columns automatically]
\end{multicols}
```

### Custom Theorem Environments
```latex
\usepackage{amsthm}
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}{Lemma}

\begin{theorem}
If $R_0 < 1$, the disease-free equilibrium is stable.
\end{theorem}
```

### Subfigures (Multiple Plots Side-by-Side)
```latex
\usepackage{subcaption}

\begin{figure}[H]
    \centering
    \begin{subfigure}{0.45\textwidth}
        \includegraphics[width=\textwidth]{plot1.png}
        \caption{Scenario A}
    \end{subfigure}
    \hfill
    \begin{subfigure}{0.45\textwidth}
        \includegraphics[width=\textwidth]{plot2.png}
        \caption{Scenario B}
    \end{subfigure}
    \caption{Comparison of two scenarios}
\end{figure}
```

---

## Common MCM/ICM Writing Pitfalls

### Pitfall 1: Over-Technical Summary Sheet
**Problem**: Using jargon like "We employ a multi-objective NSGA-II algorithm with Pareto-optimal frontiers..."
**Fix**: Simplify to "We use an optimization method to balance competing goals..."

### Pitfall 2: Undefined Variables
**Problem**: Using $\beta$ in equations without defining it first.
**Fix**: Always define variables before or immediately after first use.

### Pitfall 3: Results Without Context
**Problem**: "The optimal value is 42."
**Fix**: "The optimal facility location reduces total transportation costs by 42% compared to the baseline."

### Pitfall 4: No Sensitivity Analysis
**Problem**: Presenting results as absolute truth without testing robustness.
**Fix**: Always include a "Sensitivity Analysis" section testing key parameter variations.

---

## Final Reminder: The Summary Sheet is King

**Competition Reality**:
- Judges read 100+ papers in a short time
- If the Summary Sheet doesn't grab attention in 60 seconds, the paper is skipped
- Even a brilliant 25-page paper is worthless if the Summary Sheet fails

**Strategy**:
1. **Hour 24-36**: Draft a rough Summary Sheet outline (forces clarity of thought)
2. **Hour 60-84**: Focus on main paper (Summary Sheet outline guides writing)
3. **Hour 84-92**: Polish Summary Sheet obsessively (this is where medals are won)
4. **Hour 92-96**: Final proofread, test print, submit

**The Golden Rule**: If you only have time to polish one thing, polish the Summary Sheet.
