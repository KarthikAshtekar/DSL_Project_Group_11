# =============================================================================
#  NOTEBOOK 3 · 30-Day Technician Allocation Optimizer — Streamlit Dashboard
#
#  Run with:  streamlit run notebook3_dashboard.py
#
#  Input  : FINAL_forecast_results.xlsx   (produced by Notebook 1)
#  Solver : Gurobi (gurobipy) — requires a valid Gurobi licence
#
#  Install requirements:
#      pip install streamlit gurobipy pandas matplotlib numpy openpyxl -q
# =============================================================================

import streamlit as st
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Technician Allocation Optimizer",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    color: #e0e0f0;
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #a78bfa;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 1.5rem;
}
.dash-header {
    background: linear-gradient(90deg, rgba(167,139,250,0.15), rgba(96,165,250,0.08));
    border: 1px solid rgba(167,139,250,0.25);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
}
.dash-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.02em;
}
.dash-header p { color: #94a3b8; margin: 0; font-size: 0.9rem; }
.kpi-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
    height: 100%;
}
.kpi-card:hover { border-color: rgba(167,139,250,0.4); }
.kpi-card.green { border-color: rgba(16,185,129,0.35); }
.kpi-card.amber { border-color: rgba(245,158,11,0.35); }
.kpi-card.blue  { border-color: rgba(96,165,250,0.35); }
.kpi-label {
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
    margin-bottom: 0.4rem;
}
.kpi-value         { font-size: 2rem; font-weight: 700; color: #a78bfa; line-height: 1; }
.kpi-value.green   { color: #10b981; }
.kpi-value.amber   { color: #f59e0b; }
.kpi-value.red     { color: #ef4444; }
.kpi-value.blue    { color: #60a5fa; }
.kpi-sub           { font-size: 0.78rem; color: #475569; margin-top: 0.3rem; }
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #64748b;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding-bottom: 0.5rem;
}
.chart-panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.5rem;
}
.savings-banner {
    background: linear-gradient(90deg, rgba(16,185,129,0.12), rgba(96,165,250,0.08));
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
}
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white; border: none; border-radius: 10px;
    padding: 0.65rem 2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem; font-weight: 700;
    letter-spacing: 0.05em; width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }
.styled-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.styled-table th {
    background: rgba(167,139,250,0.15); color: #a78bfa;
    font-family: 'Space Mono', monospace; font-size: 0.68rem;
    text-transform: uppercase; letter-spacing: 0.08em;
    padding: 0.5rem 0.7rem; text-align: left;
}
.styled-table td {
    padding: 0.45rem 0.7rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    color: #e2e8f0;
}
.styled-table tr:last-child td { border-bottom: none; }
.styled-table tr:hover td { background: rgba(255,255,255,0.03); }
.badge {
    display: inline-block; padding: 0.15rem 0.5rem;
    border-radius: 999px; font-size: 0.68rem;
    font-family: 'Space Mono', monospace; font-weight: 700;
}
.badge-critical { background: rgba(239,68,68,0.2);   color: #fca5a5; }
.badge-high     { background: rgba(245,158,11,0.2);  color: #fcd34d; }
.badge-medium   { background: rgba(167,139,250,0.2); color: #c4b5fd; }
.badge-low      { background: rgba(96,165,250,0.2);  color: #93c5fd; }
.badge-vlow     { background: rgba(52,211,153,0.2);  color: #6ee7b7; }
hr { border-color: rgba(255,255,255,0.07) !important; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e", "axes.facecolor": "#1a1a2e",
    "axes.edgecolor":   "#2d2d4e", "axes.labelcolor": "#94a3b8",
    "xtick.color":      "#64748b", "ytick.color":     "#64748b",
    "text.color":       "#e0e0f0", "grid.color":      "#2d2d4e",
    "font.family":      "monospace",
})

# ── Constants ─────────────────────────────────────────────────────────────────
# Column names expected in the forecast Excel file (produced by Notebook 1)
SEVERITY_COLS_FIBRE  = ["_Critical_Fiber",    "_High_fibre",    "_Medium_fibre",
                         "_Low_fibre",          "_Very Low_fibre"]
SEVERITY_COLS_COPPER = ["new_copper_Critical", "new_copper_High", "new_copper_Medium",
                         "new_copper_Low",      "new_copper_Very Low"]
SEVERITY_LABELS      = ["Critical", "High", "Medium", "Low", "Very Low"]
SEV_COLORS           = {
    "Critical": "#ef4444", "High": "#f59e0b",
    "Medium":   "#a78bfa", "Low":  "#60a5fa", "Very Low": "#34d399",
}
BADGE_CLASSES = {
    "Critical": "badge-critical", "High": "badge-high",
    "Medium":   "badge-medium",   "Low":  "badge-low", "Very Low": "badge-vlow",
}

# Time matrix t[i, j] — hours required for severity i by technician type j
# i: 1=Critical, 2=High, 3=Medium, 4=Low, 5=Very Low
# j: 1=Intern,   2=Fresher, 3=Average, 4=Good, 5=Expert
TIME_MATRIX = {
    (1, 1): 3.0,  (1, 2): 2.6,  (1, 3): 2.5,  (1, 4): 2.2,  (1, 5): 2.0,
    (2, 1): 4.0,  (2, 2): 3.4,  (2, 3): 3.25, (2, 4): 2.8,  (2, 5): 2.5,
    (3, 1): 5.0,  (3, 2): 4.2,  (3, 3): 4.0,  (3, 4): 3.4,  (3, 5): 3.0,
    (4, 1): 6.0,  (4, 2): 5.0,  (4, 3): 4.75, (4, 4): 4.0,  (4, 5): 3.5,
    (5, 1): 7.0,  (5, 2): 5.8,  (5, 3): 5.5,  (5, 4): 4.6,  (5, 5): 4.0,
}

I = range(1, 6)   # Severity indices
J = range(1, 6)   # Technician type indices

CHARGE_PER_JOB = 10   # Revenue/saving per extra job completed vs random baseline

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <h1>🔧 30-Day Technician Allocation Optimizer</h1>
  <p>Gurobi MIP · Manual single-day OR 30-day Excel upload · Full analytics dashboard</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Forecast Data")
    uploaded_file = st.file_uploader(
        "Upload 30-day forecast Excel", type=["xlsx", "xls"],
        help="Use FINAL_forecast_results.xlsx produced by Notebook 1"
    )

    st.markdown("### ⚙️ Config")
    H = st.number_input(
        "Work hours / day", value=8, min_value=1, max_value=24,
        help="Standard shift length for all technicians"
    )

    st.markdown("### 👷 Technicians (Fixed)")
    # Workforce: {type_index: count}
    # Type 1=Intern(9), 2=Fresher(8), 3=Average(11), 4=Good(18), 5=Expert(24)
    WORKERS = {1: 9, 2: 8, 3: 11, 4: 18, 5: 24}
    st.metric("💼 Total headcount", sum(WORKERS.values()))

    st.markdown("### 🎛️ SLA Priority")
    beta = st.slider(
        "Copper weight (β)", 0.0, 1.0, 0.5, step=0.05,
        help="β = 1.0 → fibre and copper weighted equally; β < 1.0 → fibre prioritised"
    )

    st.markdown("---")
    run_30day = st.button("🚀 Run 30-Day Optimization", type="primary")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_quick, tab_30day = st.tabs(["🎯 Quick Single-Day", "📊 Full 30-Day Excel"])

# =============================================================================
#  TAB 1 — QUICK SINGLE-DAY MANUAL OPTIMISATION
# =============================================================================
with tab_quick:
    st.markdown("#### Enter today's demand across all severity levels")
    st.caption("Backlog copper = unresolved copper complaints carried forward from yesterday")

    col1, col2, col3 = st.columns(3)
    fiber_jobs, copper_jobs, backlog_copper_jobs = {}, {}, {}

    with col1:
        st.markdown("**🔵 Fibre demand**")
        for i, sev in enumerate(SEVERITY_LABELS, 1):
            fiber_jobs[i] = st.number_input(
                sev, value=15 if i <= 3 else 10, min_value=0, key=f"q_fiber_{i}"
            )

    with col2:
        st.markdown("**🟢 Copper demand**")
        for i, sev in enumerate(SEVERITY_LABELS, 1):
            copper_jobs[i] = st.number_input(
                sev, value=8, min_value=0, key=f"q_copper_{i}"
            )

    with col3:
        st.markdown("**🟡 Backlog copper**")
        for i, sev in enumerate(SEVERITY_LABELS, 1):
            backlog_copper_jobs[i] = st.number_input(
                sev, value=5 if i <= 3 else 3, min_value=0, key=f"q_blcopper_{i}"
            )

    if st.button("🧮 Optimize Now", type="primary"):
        F_demand = {i: fiber_jobs[i] for i in I}
        C_demand = {i: copper_jobs[i] + backlog_copper_jobs[i] for i in I}

        # ── Gurobi MIP (single day) ───────────────────────────────────────────
        m = gp.Model("single_day")
        m.Params.OutputFlag = 0

        xF = m.addVars(I, J, vtype=GRB.INTEGER, lb=0, name="Fibre")
        xC = m.addVars(I, J, vtype=GRB.INTEGER, lb=0, name="Copper")

        # Objective: maximise jobs completed (copper weighted by β)
        m.setObjective(
            gp.quicksum(xF[i, j] for i in I for j in J)
            + beta * gp.quicksum(xC[i, j] for i in I for j in J),
            GRB.MAXIMIZE,
        )

        # Constraint 1 — Demand feasibility
        for i in I:
            m.addConstr(gp.quicksum(xF[i, j] for j in J) <= F_demand[i])
            m.addConstr(gp.quicksum(xC[i, j] for j in J) <= C_demand[i])

        # Constraint 2 — Capacity per technician type
        for j in J:
            m.addConstr(
                gp.quicksum(TIME_MATRIX[i, j] * (xF[i, j] + xC[i, j]) for i in I)
                <= WORKERS[j] * H
            )
            # Constraint 3 — Copper SLA 48-hour time window (2× standard capacity)
            m.addConstr(
                gp.quicksum(TIME_MATRIX[i, j] * xC[i, j] for i in I)
                <= 2 * WORKERS[j] * H
            )

        m.optimize()

        if m.Status == GRB.OPTIMAL:
            opt_fibre  = {i: int(sum(xF[i, j].X for j in J)) for i in I}
            opt_copper = {i: int(sum(xC[i, j].X for j in J)) for i in I}

            fibre_done  = sum(opt_fibre.values())
            copper_done = sum(opt_copper.values())
            fibre_bl    = sum(F_demand.values()) - fibre_done
            copper_bl   = sum(C_demand.values()) - copper_done

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("✅ Fibre Completed",  fibre_done,  delta=f"-{fibre_bl} backlog")
            col_b.metric("✅ Copper Completed", copper_done, delta=f"-{copper_bl} backlog")
            col_c.metric("🏁 Total Jobs",       fibre_done + copper_done)

            results_tbl = pd.DataFrame({
                "Severity":      SEVERITY_LABELS,
                "Fibre Demand":  [F_demand[i]  for i in I],
                "Fibre Done":    [opt_fibre[i]  for i in I],
                "Copper Demand": [C_demand[i]  for i in I],
                "Copper Done":   [opt_copper[i] for i in I],
            })
            st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
            st.markdown("**Allocation Results**")
            st.dataframe(
                results_tbl.style.format({"Fibre Done": "{:.0f}", "Copper Done": "{:.0f}"}),
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("⚠️  Gurobi did not find an optimal solution. Check demand inputs.")

# =============================================================================
#  TAB 2 — FULL 30-DAY ROLLING OPTIMISATION
# =============================================================================
with tab_30day:
    st.markdown('<div class="section-title">Upload forecast Excel in sidebar → Click Run</div>',
                unsafe_allow_html=True)

    if run_30day:
        if uploaded_file is None:
            st.warning("⬅️  Please upload the forecast Excel file in the sidebar first.")
            st.stop()

        # ── Load and validate Excel ───────────────────────────────────────────
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as exc:
            st.error(f"❌ Failed to read Excel: {exc}")
            st.stop()

        required_cols = (["Forecasted_day_index"]
                         + SEVERITY_COLS_FIBRE + SEVERITY_COLS_COPPER)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"❌ Missing columns: {missing}")
            st.stop()

        df = df[required_cols].copy()
        df["Forecasted_day_index"] = df["Forecasted_day_index"].astype(int)
        df = df.sort_values("Forecasted_day_index").reset_index(drop=True)

        for col in SEVERITY_COLS_FIBRE + SEVERITY_COLS_COPPER:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        num_days = len(df)

        # ── 30-day rolling optimisation ───────────────────────────────────────
        with st.spinner(f"Running Gurobi over {num_days} days…"):

            daily_results  = []
            fibre_backlog  = {i: 0 for i in I}
            copper_backlog = {i: 0 for i in I}
            rand_fibre_bl  = {i: 0 for i in I}
            rand_copper_bl = {i: 0 for i in I}

            for _, row in df.iterrows():
                day_num = int(row["Forecasted_day_index"])

                # Effective demand = fresh forecast + previous day backlog
                F_demand = {
                    i: int(row[SEVERITY_COLS_FIBRE[i - 1]]) + fibre_backlog[i]
                    for i in I
                }
                C_demand = {
                    i: int(row[SEVERITY_COLS_COPPER[i - 1]]) + copper_backlog[i]
                    for i in I
                }

                # ── Gurobi MIP ────────────────────────────────────────────────
                mdl = gp.Model(f"Day_{day_num}")
                mdl.setParam("OutputFlag", 0)

                xF = mdl.addVars(I, J, vtype=GRB.INTEGER, lb=0, name="Fibre")
                xC = mdl.addVars(I, J, vtype=GRB.INTEGER, lb=0, name="Copper")

                mdl.setObjective(
                    gp.quicksum(xF[i, j] for i in I for j in J)
                    + beta * gp.quicksum(xC[i, j] for i in I for j in J),
                    GRB.MAXIMIZE,
                )

                for i in I:
                    mdl.addConstr(
                        gp.quicksum(xF[i, j] for j in J) <= F_demand[i],
                        f"FibreDemand_{i}",
                    )
                    mdl.addConstr(
                        gp.quicksum(xC[i, j] for j in J) <= C_demand[i],
                        f"CopperDemand_{i}",
                    )

                for j in J:
                    mdl.addConstr(
                        gp.quicksum(
                            TIME_MATRIX[i, j] * (xF[i, j] + xC[i, j]) for i in I
                        ) <= WORKERS[j] * H,
                        f"Capacity_{j}",
                    )
                    mdl.addConstr(
                        gp.quicksum(TIME_MATRIX[i, j] * xC[i, j] for i in I)
                        <= 2 * WORKERS[j] * H,
                        f"CopperWindow_{j}",
                    )

                mdl.optimize()

                if mdl.Status == GRB.OPTIMAL:
                    opt_fibre  = {i: int(sum(xF[i, j].X for j in J)) for i in I}
                    opt_copper = {i: int(sum(xC[i, j].X for j in J)) for i in I}
                else:
                    opt_fibre  = {i: 0 for i in I}
                    opt_copper = {i: 0 for i in I}

                total_opt_fibre  = sum(opt_fibre.values())
                total_opt_copper = sum(opt_copper.values())
                total_opt_jobs   = total_opt_fibre + total_opt_copper

                # Per-severity backlogs carried to next day
                new_fibre_bl  = {i: max(0, F_demand[i] - opt_fibre[i])  for i in I}
                new_copper_bl = {i: max(0, C_demand[i] - opt_copper[i]) for i in I}

                sev_backlog_today = {
                    SEVERITY_LABELS[i - 1]: new_fibre_bl[i] + new_copper_bl[i]
                    for i in I
                }
                fibre_backlog_sev  = {SEVERITY_LABELS[i - 1]: new_fibre_bl[i]  for i in I}
                copper_backlog_sev = {SEVERITY_LABELS[i - 1]: new_copper_bl[i] for i in I}
                total_backlog_today = sum(sev_backlog_today.values())

                # ── Random baseline (proportional capacity split) ──────────────
                total_cap_hours  = sum(WORKERS[j] * H for j in J)
                avg_job_time     = np.mean(list(TIME_MATRIX.values()))
                rand_cap_jobs    = int(total_cap_hours / avg_job_time)

                rand_F_demand    = {
                    i: int(row[SEVERITY_COLS_FIBRE[i - 1]]) + rand_fibre_bl[i]
                    for i in I
                }
                rand_C_demand    = {
                    i: int(row[SEVERITY_COLS_COPPER[i - 1]]) + rand_copper_bl[i]
                    for i in I
                }
                rand_total_dem   = (sum(rand_F_demand.values())
                                    + sum(rand_C_demand.values()))
                rand_jobs        = min(rand_cap_jobs, rand_total_dem)
                rand_unmet       = max(0, rand_total_dem - rand_jobs)

                if rand_total_dem > 0:
                    for i in I:
                        share        = ((rand_F_demand[i] + rand_C_demand[i])
                                        / rand_total_dem)
                        unmet_i      = int(rand_unmet * share)
                        denom        = rand_F_demand[i] + rand_C_demand[i] + 1e-9
                        rand_fibre_bl[i]  = int(unmet_i * rand_F_demand[i] / denom)
                        rand_copper_bl[i] = unmet_i - rand_fibre_bl[i]
                else:
                    rand_fibre_bl  = {i: 0 for i in I}
                    rand_copper_bl = {i: 0 for i in I}

                rand_total_bl = (sum(rand_fibre_bl.values())
                                 + sum(rand_copper_bl.values()))

                daily_results.append({
                    "day":               day_num,
                    "opt_fibre":         total_opt_fibre,
                    "opt_copper":        total_opt_copper,
                    "opt_jobs":          total_opt_jobs,
                    "opt_backlog":       total_backlog_today,
                    "sev_backlog":       sev_backlog_today,
                    "fibre_backlog_sev": fibre_backlog_sev,
                    "copper_backlog_sev":copper_backlog_sev,
                    "rand_jobs":         rand_jobs,
                    "rand_backlog":      rand_total_bl,
                    "extra_jobs":        max(0, total_opt_jobs - rand_jobs),
                    "utilization": [
                        (sum(TIME_MATRIX[i, j] * (opt_fibre[i] + opt_copper[i])
                             for i in I) / (WORKERS[j] * H))
                        if WORKERS[j] * H > 0 else 0
                        for j in J
                    ],
                })

                fibre_backlog  = new_fibre_bl
                copper_backlog = new_copper_bl

        results_df = pd.DataFrame(daily_results)

        # ── Aggregate KPIs ────────────────────────────────────────────────────
        total_opt_month  = int(results_df["opt_jobs"].sum())
        total_rand_month = int(results_df["rand_jobs"].sum())
        total_extra      = int(results_df["extra_jobs"].sum())
        monthly_savings  = total_extra * CHARGE_PER_JOB
        annual_savings   = monthly_savings * 12
        avg_bl           = results_df["opt_backlog"].mean()
        peak_bl_idx      = results_df["opt_backlog"].idxmax()
        peak_bl_day      = results_df.loc[peak_bl_idx, "day"]
        peak_bl_val      = int(results_df.loc[peak_bl_idx, "opt_backlog"])
        avg_util         = float(np.mean([np.mean(r) for r in results_df["utilization"]]))
        days_arr         = results_df["day"].values

        # 30-day per-severity backlog totals
        sev_bl_totals = {
            sev: int(sum(r["sev_backlog"][sev] for _, r in results_df.iterrows()))
            for sev in SEVERITY_LABELS
        }

        # ── Savings banner ────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="savings-banner">
            <div style="display:flex; gap:3rem; align-items:center; flex-wrap:wrap;">
                <div>
                    <div style="font-family:'Space Mono',monospace; font-size:0.68rem;
                         text-transform:uppercase; letter-spacing:0.12em;
                         color:#6ee7b7; margin-bottom:0.3rem;">Monthly Cost Savings</div>
                    <div style="font-size:2.4rem; font-weight:700; color:#10b981;">
                        ${monthly_savings:,}</div>
                    <div style="font-size:0.8rem; color:#475569; margin-top:0.2rem;">
                        {total_extra:,} extra jobs × ${CHARGE_PER_JOB}/job</div>
                </div>
                <div style="width:1px; background:rgba(255,255,255,0.1); height:60px;"></div>
                <div>
                    <div style="font-family:'Space Mono',monospace; font-size:0.68rem;
                         text-transform:uppercase; letter-spacing:0.12em;
                         color:#93c5fd; margin-bottom:0.3rem;">Annual Projection</div>
                    <div style="font-size:2.4rem; font-weight:700; color:#60a5fa;">
                        ${annual_savings:,}</div>
                    <div style="font-size:0.8rem; color:#475569;">Monthly × 12</div>
                </div>
                <div style="width:1px; background:rgba(255,255,255,0.1); height:60px;"></div>
                <div>
                    <div style="font-family:'Space Mono',monospace; font-size:0.68rem;
                         text-transform:uppercase; letter-spacing:0.12em;
                         color:#fbbf24; margin-bottom:0.3rem;">Extra Jobs vs Random</div>
                    <div style="font-size:2.4rem; font-weight:700; color:#f59e0b;">
                        +{total_extra:,}</div>
                    <div style="font-size:0.8rem; color:#475569;">Optimizer vs baseline</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── KPI cards ─────────────────────────────────────────────────────────
        st.markdown('<div class="section-title">30-Day Summary KPIs</div>',
                    unsafe_allow_html=True)

        def kpi(col, label, value, sub="", val_class="", card_class=""):
            col.markdown(f"""
            <div class="kpi-card {card_class}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value {val_class}">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        kpi(k1, "Total Jobs (Optimizer)", f"{total_opt_month:,}",
            "30-day total", "green", "green")
        kpi(k2, "Total Jobs (Random)",    f"{total_rand_month:,}",
            "non-data-driven", "amber")
        kpi(k3, "Improvement",            f"+{total_extra:,}",
            "extra jobs", "green")
        kpi(k4, "Avg Daily Backlog",      f"{avg_bl:.1f}",
            "jobs carried fwd/day", "amber", "amber")
        kpi(k5, "Peak Backlog",           str(peak_bl_val),
            f"on day {peak_bl_day}", "red")
        kpi(k6, "Avg Utilization",        f"{avg_util:.0%}",
            "across all tech types", "blue", "blue")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Chart 1 — Daily throughput comparison ─────────────────────────────
        st.markdown('<div class="section-title">Daily Performance</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns([1.6, 1])

        with c1:
            st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
            st.markdown("**Optimizer vs Random — Daily Jobs Completed**")
            fig1, ax1 = plt.subplots(figsize=(9, 4))
            ax1.fill_between(
                days_arr, results_df["rand_jobs"], results_df["opt_jobs"],
                alpha=0.25, color="#10b981", label="Extra jobs (gain)",
            )
            ax1.plot(days_arr, results_df["opt_jobs"],  color="#10b981",
                     lw=2, label="Optimizer", zorder=3)
            ax1.plot(days_arr, results_df["rand_jobs"], color="#f59e0b",
                     lw=1.5, linestyle="--", label="Random", zorder=3)
            ax1.set_xlabel("Day"); ax1.set_ylabel("Jobs Completed")
            ax1.set_title("Daily Throughput Comparison", color="#94a3b8", pad=10)
            ax1.legend(fontsize=8, framealpha=0.2)
            ax1.grid(axis="y", zorder=0)
            ax1.spines[["top", "right"]].set_visible(False)
            fig1.tight_layout()
            st.pyplot(fig1)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
            st.markdown("**Daily Total Backlog Carryforward**")
            fig2, ax2 = plt.subplots(figsize=(5, 3.8))
            ax2.bar(days_arr, results_df["opt_backlog"],
                    color="#ef4444", width=0.6, label="Optimizer backlog",
                    alpha=0.85, zorder=3)
            ax2.bar(days_arr, results_df["rand_backlog"],
                    color="#f59e0b", width=0.6, label="Random backlog",
                    alpha=0.40, zorder=3)
            ax2.set_xlabel("Day"); ax2.set_ylabel("Backlog Jobs")
            ax2.set_title("Unmet Jobs → Next Day", color="#94a3b8", pad=10)
            ax2.legend(fontsize=8, framealpha=0.2)
            ax2.grid(axis="y", zorder=0)
            ax2.spines[["top", "right"]].set_visible(False)
            fig2.tight_layout()
            st.pyplot(fig2)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Chart 2 — Per-severity backlog breakdown ──────────────────────────
        st.markdown('<div class="section-title">Backlog by Severity Category</div>',
                    unsafe_allow_html=True)
        bs1, bs2 = st.columns([1.6, 1])

        sev_daily = {
            sev: [r["sev_backlog"][sev] for _, r in results_df.iterrows()]
            for sev in SEVERITY_LABELS
        }

        with bs1:
            st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
            st.markdown("**Daily Backlog Stacked by Severity (Critical → Very Low)**")
            fig3, ax3 = plt.subplots(figsize=(9, 3.8))
            bottoms = np.zeros(len(days_arr))
            for sev in SEVERITY_LABELS:
                vals = np.array(sev_daily[sev])
                ax3.bar(days_arr, vals, bottom=bottoms, width=0.6,
                        label=sev, color=SEV_COLORS[sev], alpha=0.9, zorder=3)
                bottoms += vals
            ax3.set_xlabel("Day"); ax3.set_ylabel("Backlog Jobs")
            ax3.set_title("Daily Backlog Split by Severity", color="#94a3b8", pad=10)
            ax3.legend(fontsize=8, framealpha=0.2, loc="upper right")
            ax3.grid(axis="y", zorder=0)
            ax3.spines[["top", "right"]].set_visible(False)
            fig3.tight_layout()
            st.pyplot(fig3)
            st.markdown("</div>", unsafe_allow_html=True)

        with bs2:
            st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
            st.markdown("**30-Day Backlog Distribution by Severity**")

            sev_vals = [sev_bl_totals[sev] for sev in SEVERITY_LABELS]
            if sum(sev_vals) > 0:
                fig4, ax4 = plt.subplots(figsize=(5, 4))
                wedges, texts, autotexts = ax4.pie(
                    sev_vals,
                    labels=SEVERITY_LABELS,
                    colors=[SEV_COLORS[s] for s in SEVERITY_LABELS],
                    autopct="%1.0f%%",
                    startangle=140,
                    pctdistance=0.8,
                    labeldistance=1.1,
                    textprops={"fontsize": 9},
                    wedgeprops=dict(width=0.6, edgecolor="#1a1a2e", linewidth=1.5),
                )
                for at in autotexts:
                    at.set_fontsize(9); at.set_color("white")
                ax4.set_title("30-Day Backlog Share", color="#94a3b8", pad=10)
                fig4.tight_layout()
                st.pyplot(fig4)
            else:
                st.success("✅ Zero backlog across all severities!")

            # Per-severity summary table
            rows_html = "".join(
                f'<tr>'
                f'<td><span class="badge {BADGE_CLASSES[sev]}">{sev}</span></td>'
                f'<td>{sev_bl_totals[sev]:,}</td>'
                f'<td>{sev_bl_totals[sev] / num_days:.1f}</td>'
                f'</tr>'
                for sev in SEVERITY_LABELS
            )
            st.markdown(f"""
            <table class="styled-table" style="margin-top:0.8rem;">
                <thead>
                    <tr><th>Severity</th><th>Total BL</th><th>Avg/Day</th></tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#475569; padding:2rem; font-size:0.8rem;">
    Intelligent Technician Allocation System · Gurobi MIP Optimizer ·
    Notebook 3 of 3
</div>
""", unsafe_allow_html=True)
