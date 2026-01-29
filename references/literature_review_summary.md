# Literature Review Summary: MCM 2025 Problem B (Juneau Sustainable Tourism)

## 1. System Dynamics & Multi-Objective Optimization (NSGA-II)
**Paper**: *Overtourism to Equilibrium: A System Dynamics & Multi-Objective Model for Sustainable Destinations*
**Authors**: Lyu, H., Yang, X., & Ji, X. (2025)
**URL**: https://arxiv.org/abs/2511.14288

### Core Equations / Logic:
- **Objective Function**: Max sum (Net Revenue) - lambda_1 * (Env Impact) + lambda_2 * (Social Satisfaction)
- **Glacier Recession Model**: Delta G = f(T, CO2, V) where V is visitor numbers (indirectly through carbon footprint).
- **Social Well-being**: S = f(V, R) where R is government expenditure on local services.

### State Variables:
- V(t): Annual visitor arrivals (Stock)
- G(t): Glacier length/area (Stock)
- M(t): Total CO2 emissions
- U(t): Social satisfaction index

### Key Parameters:
- **Revenue per Tourist**: Economic yield.
- **Emission Factor**: Tons of CO2 per cruise passenger.
- **Crowding Sensitivity**: Marginal decrease in satisfaction per additional 1,000 tourists.

---

## 2. Tourism Carrying Capacity (TCC) Mathematical Framework
**Paper**: *The Estimation of Physical and Real Carrying Capacity with Application on Tourist Sites*
**Authors**: Cifuentes, M. (1992); Attallah, N. F. (2021)
**URL**: http://indexing.jotr.eu/Jotr/Volume12/V12-5.pdf

### Core Equations:
1. **Physical Carrying Capacity (PCC)**:
   PCC = A * (V/a) * Rf
   - A: Available area for tourism.
   - V/a: Area required per person.
   - Rf: Rotation factor (opening hours / visit duration).
2. **Real Carrying Capacity (RCC)**:
   RCC = PCC * (Cf_1 * Cf_2 * ... * Cf_n)
   - Cf_i: Correction factors (social, environmental, accessibility thresholds).
3. **Effective Carrying Capacity (ECC)**:
   ECC = RCC * MC
   - MC: Management Capacity (staffing, infrastructure efficiency).

### Key Parameters:
- **Space standard**: m2 per visitor.
- **Visit duration**: average time spent at Mendenhall Glacier.
- **Limiting factors**: Rain/snow frequency, staff-to-visitor ratio.

---

## 3. Dynamic Model with Resident Spillovers
**Paper**: *Sustainable tourism development: A dynamic model incorporating resident spillovers*
**Authors**: Schubert, S. F., & Schamel, G. (2020)
**URL**: https://doi.org/10.1177/1354816620934552

### Core Equations:
- **Resident Utility**: U_t = integral e^(-rho t) [u(C_t, Q_t, N_t)] dt
  - C: Private consumption.
  - Q: Tourism-induced quality of life (e.g., better restaurants, infrastructure).
  - N: Number of tourists (congestion/negative spillover).
- **Tourism Demand**: N_t = f(P_t, Q_t) where P is price/tax.

### State Variables:
- **Tourism Capital (K)**: Infrastructure and facilities.
- **Resident Well-being (W)**: Accumulated utility/satisfaction.

### Key Parameters:
- **rho**: Discount rate of residents for future benefits.
- **Elasticity of substitution**: Between private consumption and public tourism quality.

---

## 4. Pressure-State-Response (PSR) Evaluation Model
**Paper**: *Towards management of sustainable tourism development: insights from a tourism carrying capacity analysis*
**Authors**: Xu, N., & Li, H. (2025)
**URL**: https://link.springer.com/article/10.1007/s43621-025-00951-1

### Core Equations:
- **Normalization (Min-Max)**: x_ij* = (x_ij - min(x_ij)) / (max(x_ij) - min(x_ij))
- **Entropy Weighting**: w_j calculated based on data divergence.
- **Composite Index**: TCC_i = sum w_j * x_ij*

### Modules:
- **Pressure**: Tourist arrivals, Receipts, Tourism density.
- **State**: Sea water quality, Air quality, Resource availability.
- **Response**: Waste treatment rate, Green coverage, Policy effectiveness.

---

## 5. Juneau Specific Context & Management
**Report/Agreement**: *Juneau Cruise Passenger Cap Agreement (2024)* & *TBMP Guidelines*
**Source**: City and Borough of Juneau (CBJ)
**URL**: https://juneau.org/manager/tbmp

### Key Specifics:
- **Daily Cap (2026)**: 16,000 passengers (Sun-Fri), 12,000 (Sat).
- **Historical Peak**: Up to 21,000 visitors/day.
- **Mendenhall Glacier Limit**: US Forest Service manages specific commercial use permits (approx. 500,000/year cap on glacier area visitors).
- **TBMP**: Focuses on noise (helicopter paths), emissions (shore power), and traffic (idling).

### Implications for Modeling:
- Models should use a **Constraint-based approach** where V_max is a hard limit.
- **Seasonality** is critical: May to September (approx. 150 days).
