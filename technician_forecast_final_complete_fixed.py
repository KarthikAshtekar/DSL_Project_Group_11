import streamlit as st
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import io

st.set_page_config(page_title="30-Day Technician Optimizer", page_icon="🔧", layout="wide", initial_sidebar_state="expanded")

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
.insight-card {
    background: rgba(255,255,255,0.04);
    border-left: 3px solid #a78bfa;
    border-radius: 0 10px 10px 0;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.88rem;
    color: #cbd5e1;
}
.insight-card.warn  { border-left-color: #f59e0b; }
.insight-card.error { border-left-color: #ef4444; }
.insight-card.ok    { border-left-color: #10b981; }
.insight-card.info  { border-left-color: #60a5fa; }
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
.badge-ok       { background: rgba(16,185,129,0.15);  color: #10b981; }
.badge-warn     { background: rgba(245,158,11,0.15);  color: #f59e0b; }
.badge-error    { background: rgba(239,68,68,0.15);   color: #ef4444; }
.badge-blue     { background: rgba(96,165,250,0.15);  color: #60a5fa; }
.badge-critical { background: rgba(239,68,68,0.2);    color: #fca5a5; }
.badge-high     { background: rgba(245,158,11,0.2);   color: #fcd34d; }
.badge-medium   { background: rgba(167,139,250,0.2);  color: #c4b5fd; }
.badge-low      { background: rgba(96,165,250,0.2);   color: #93c5fd; }
.badge-vlow     { background: rgba(52,211,153,0.2);   color: #6ee7b7; }
hr { border-color: rgba(255,255,255,0.07) !important; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "#1a1a2e", "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "#2d2d4e", "axes.labelcolor": "#94a3b8",
    "xtick.color": "#64748b", "ytick.color": "#64748b",
    "text.color": "#e0e0f0", "grid.color": "#2d2d4e", "font.family": "monospace"
})

SEVERITY_COLS_FIBRE  = ["_Critical_Fiber", "_High_fibre", "_Medium_fibre", "_Low_fibre", "_Very Low_fibre"]
SEVERITY_COLS_COPPER = ["new_copper_Critical", "new_copper_High", "new_copper_Medium", "new_copper_Low", "new_copper_Very Low"]
SEVERITY_LABELS = ["Critical", "High", "Medium", "Low", "Very Low"]
SEV_COLORS = {"Critical": "#ef4444", "High": "#f59e0b", "Medium": "#a78bfa", "Low": "#60a5fa", "Very Low": "#34d399"}

t = {
    (1,1):3.0,(1,2):2.6,(1,3):2.5,(1,4):2.2,(1,5):2.0,
    (2,1):4.0,(2,2):3.4,(2,3):3.25,(2,4):2.8,(2,5):2.5,
    (3,1):5.0,(3,2):4.2,(3,3):4.0,(3,4):3.4,(3,5):3.0,
    (4,1):6.0,(4,2):5.0,(4,3):4.75,(4,4):4.0,(4,5):3.5,
    (5,1):7.0,(5,2):5.8,(5,3):5.5,(5,4):4.6,(5,5):4.0,
}
I = range(1, 6)
J = range(1, 6)
CHARGE_PER_JOB = 10

st.markdown("""
<div class="dash-header">
  <h1>🔧 30-Day Technician Allocation Optimizer</h1>
  <p>Gurobi-powered · Manual OR Excel · Complete dashboard</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📂 Forecast Data")
    uploaded_file = st.file_uploader("Upload 30-day Excel", type=["xlsx", "xls"])
    
    st.markdown("### ⚙️ Config")
    H = st.number_input("Work hours/day", value=8, min_value=1, max_value=24)
    
    st.markdown("### 👷 Technicians (Fixed)")
    workers = {1:9, 2:8, 3:11, 4:18, 5:24}
    st.metric("💼 Total", sum(workers.values()))
    
    st.markdown("### 🎛️ Priority")
    beta = st.slider("Copper Weight (β)", 0.0, 1.0, 0.5)
    
    st.markdown("---")
    run_30day = st.button("🚀 Run 30-Day Optimization", type="primary")

tab_quick, tab_30day = st.tabs(["🎯 Quick Manual Single Day", "📊 Full 30-Day Excel"])

with tab_quick:
    st.markdown("**Enter current demand** (Fiber, Copper, Backlog Copper)")
    
    col1, col2, col3 = st.columns(3)
    fiber_jobs = {}
    copper_jobs = {}
    backlog_copper_jobs = {}
    
    with col1:
        st.markdown("**🔵 Fiber**")
        for i, sev in enumerate(SEVERITY_LABELS,1):
            fiber_jobs[i] = st.number_input(sev, value=10 if i>3 else 15, min_value=0, key=f"q_fiber_{i}")
    
    with col2:
        st.markdown("**🟢 Copper**")
        for i, sev in enumerate(SEVERITY_LABELS,1):
            copper_jobs[i] = st.number_input(sev, value=8, min_value=0, key=f"q_copper_{i}")
    
    with col3:
        st.markdown("**🟡 Backlog Copper**")
        for i, sev in enumerate(SEVERITY_LABELS,1):
            backlog_copper_jobs[i] = st.number_input(sev, value=3 if i>3 else 5, min_value=0, key=f"q_blcopper_{i}")
    
    if st.button("🧮 Optimize Now", type="primary"):
        C_demand = {i: copper_jobs[i] + backlog_copper_jobs[i] for i in I}
        F_demand = {i: fiber_jobs[i] for i in I}
        
        model = gp.Model("quick_opt")
        model.Params.OutputFlag = 0
        xF = model.addVars(I, J, vtype=GRB.INTEGER)
        xC = model.addVars(I, J, vtype=GRB.INTEGER)
        
        model.setObjective(gp.quicksum(xF[i,j] for i in I for j in J) + beta * gp.quicksum(xC[i,j] for i in I for j in J), GRB.MAXIMIZE)
        
        for i in I:
            model.addConstr(gp.quicksum(xF[i,j] for j in J) <= F_demand[i])
            model.addConstr(gp.quicksum(xC[i,j] for j in J) <= C_demand[i])
        for j in J:
            model.addConstr(gp.quicksum(t[i,j]*(xF[i,j]+xC[i,j]) for i in I) <= workers[j]*H)
            model.addConstr(gp.quicksum(t[i,j]*xC[i,j] for i in I) <= 2*workers[j]*H)
        
        model.optimize()
        
        if model.Status == GRB.OPTIMAL:
            opt_fibre = {i: sum(xF[i,j].X for j in J) for i in I}
            opt_copper = {i: sum(xC[i,j].X for j in J) for i in I}
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Fiber Completed", sum(opt_fibre.values()), delta=f"-{sum(F_demand.values())-sum(opt_fibre.values())} backlog")
            with col_b:
                st.metric("Copper Completed", sum(opt_copper.values()), delta=f"-{sum(C_demand.values())-sum(opt_copper.values())} backlog")
            
            df_results = pd.DataFrame({
                'Severity': SEVERITY_LABELS,
                'Fiber Demand': [F_demand[i] for i in I],
                'Fiber Done': [opt_fibre[i] for i in I],
                'Copper Demand': [C_demand[i] for i in I],
                'Copper Done': [opt_copper[i] for i in I]
            })
            st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
            st.markdown("**Results Table**")
            st.dataframe(df_results.style.format({'Fiber Done': '{:.0f}', 'Copper Done': '{:.0f}'}))
            st.markdown('</div>', unsafe_allow_html=True)

with tab_30day:
    st.markdown('<div class="section-title">Upload Excel in Sidebar → Click Run Button</div>', unsafe_allow_html=True)
    
    if run_30day and uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"❌ Failed to read Excel: {e}")
            st.stop()

        required_cols = ["Forecasted_day_index"] + SEVERITY_COLS_FIBRE + SEVERITY_COLS_COPPER
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"❌ Missing columns in Excel: {missing}")
            st.stop()

        df = df[required_cols].copy()
        df["Forecasted_day_index"] = df["Forecasted_day_index"].astype(int)
        df = df.sort_values("Forecasted_day_index").reset_index(drop=True)

        for col in SEVERITY_COLS_FIBRE + SEVERITY_COLS_COPPER:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        num_days = len(df)

        with st.spinner(f"Running Gurobi optimizer across {num_days} days..."):

            daily_results = []
            fibre_backlog  = {i: 0 for i in I}
            copper_backlog = {i: 0 for i in I}
            random_fibre_backlog  = {i: 0 for i in I}
            random_copper_backlog = {i: 0 for i in I}

            for day_idx, row in df.iterrows():
                day_num = int(row["Forecasted_day_index"])

                # Demand = fresh forecast + carryover backlog
                F_demand = {i: int(row[SEVERITY_COLS_FIBRE[i-1]])  + fibre_backlog[i]  for i in I}
                C_demand = {i: int(row[SEVERITY_COLS_COPPER[i-1]]) + copper_backlog[i] for i in I}

                # ── Gurobi ──────────────────────────────────────────────────
                model = gp.Model(f"Day_{day_num}")
                model.setParam("OutputFlag", 0)

                xF = model.addVars(I, J, vtype=GRB.INTEGER, lb=0, name="Fibre")
                xC = model.addVars(I, J, vtype=GRB.INTEGER, lb=0, name="Copper")

                model.setObjective(
                    gp.quicksum(xF[i,j] for i in I for j in J)
                    + beta * gp.quicksum(xC[i,j] for i in I for j in J),
                    GRB.MAXIMIZE
                )

                for i in I:
                    model.addConstr(gp.quicksum(xF[i,j] for j in J) <= F_demand[i], f"FibreDemand_{i}")
                    model.addConstr(gp.quicksum(xC[i,j] for j in J) <= C_demand[i], f"CopperDemand_{i}")

                for j in J:
                    model.addConstr(
                        gp.quicksum(t[i,j] * (xF[i,j] + xC[i,j]) for i in I) <= workers[j] * H,
                        f"Capacity_{j}"
                    )
                    model.addConstr(
                        gp.quicksum(t[i,j] * xC[i,j] for i in I) <= 2 * workers[j] * H,
                        f"CopperWindow_{j}"
                    )

                model.optimize()

                if model.status == GRB.OPTIMAL:
                    opt_fibre  = {i: sum(int(xF[i,j].X) for j in J) for i in I}
                    opt_copper = {i: sum(int(xC[i,j].X) for j in J) for i in I}
                else:
                    opt_fibre  = {i: 0 for i in I}
                    opt_copper = {i: 0 for i in I}

                total_opt_fibre  = sum(opt_fibre.values())
                total_opt_copper = sum(opt_copper.values())
                total_opt_jobs   = total_opt_fibre + total_opt_copper

                # ── Per-severity backlog ─────────────────────────────────────
                new_fibre_backlog  = {i: max(0, F_demand[i] - opt_fibre[i])  for i in I}
                new_copper_backlog = {i: max(0, C_demand[i] - opt_copper[i]) for i in I}

                # Combined per-severity (fibre + copper) for display
                sev_backlog_today = {
                    SEVERITY_LABELS[i-1]: new_fibre_backlog[i] + new_copper_backlog[i]
                    for i in I
                }
                # Fibre-only and copper-only per severity for split chart
                fibre_backlog_sev  = {SEVERITY_LABELS[i-1]: new_fibre_backlog[i]  for i in I}
                copper_backlog_sev = {SEVERITY_LABELS[i-1]: new_copper_backlog[i] for i in I}

                total_backlog_today = sum(sev_backlog_today.values())

                # ── Random baseline ──────────────────────────────────────────
                total_capacity_hours     = sum(workers[j] * H for j in J)
                avg_job_time             = np.mean(list(t.values()))
                rand_total_capacity_jobs = int(total_capacity_hours / avg_job_time)

                rand_F_demand     = {i: int(row[SEVERITY_COLS_FIBRE[i-1]])  + random_fibre_backlog[i]  for i in I}
                rand_C_demand     = {i: int(row[SEVERITY_COLS_COPPER[i-1]]) + random_copper_backlog[i] for i in I}
                rand_total_demand = sum(rand_F_demand.values()) + sum(rand_C_demand.values())
                rand_jobs         = min(rand_total_capacity_jobs, rand_total_demand)

                rand_unmet = max(0, rand_total_demand - rand_jobs)
                if rand_total_demand > 0:
                    for i in I:
                        share        = (rand_F_demand[i] + rand_C_demand[i]) / rand_total_demand
                        rand_unmet_i = int(rand_unmet * share)
                        random_fibre_backlog[i]  = int(rand_unmet_i * rand_F_demand[i] / (rand_F_demand[i] + rand_C_demand[i] + 1e-9))
                        random_copper_backlog[i] = rand_unmet_i - random_fibre_backlog[i]
                else:
                    random_fibre_backlog  = {i: 0 for i in I}
                    random_copper_backlog = {i: 0 for i in I}

                rand_total_backlog = sum(random_fibre_backlog.values()) + sum(random_copper_backlog.values())

                daily_results.append({
                    "day":                 day_num,
                    "fibre_demand_fresh":  sum(int(row[SEVERITY_COLS_FIBRE[i-1]])  for i in I),
                    "copper_demand_fresh": sum(int(row[SEVERITY_COLS_COPPER[i-1]]) for i in I),
                    "fibre_demand_total":  sum(F_demand.values()),
                    "copper_demand_total": sum(C_demand.values()),
                    "opt_fibre":           total_opt_fibre,
                    "opt_copper":          total_opt_copper,
                    "opt_jobs":            total_opt_jobs,
                    "opt_backlog":         total_backlog_today,
                    "sev_backlog":         sev_backlog_today,       # combined per-severity
                    "fibre_backlog_sev":   fibre_backlog_sev,       # fibre-only per-severity
                    "copper_backlog_sev":  copper_backlog_sev,      # copper-only per-severity
                    "rand_jobs":           rand_jobs,
                    "rand_backlog":        rand_total_backlog,
                    "extra_jobs":          max(0, total_opt_jobs - rand_jobs),
                    "utilization":         [
                        sum(t[i,j]*(opt_fibre[i]+opt_copper[i]) for i in I) / (workers[j]*H)
                        if workers[j]*H > 0 else 0
                        for j in J
                    ],
                })

                fibre_backlog  = new_fibre_backlog
                copper_backlog = new_copper_backlog

            results_df = pd.DataFrame(daily_results)

        # ─────────────────────────────────────────────
        # AGGREGATE METRICS
        # ─────────────────────────────────────────────
        total_opt_jobs_month  = results_df["opt_jobs"].sum()
        total_rand_jobs_month = results_df["rand_jobs"].sum()
        total_extra_jobs      = results_df["extra_jobs"].sum()
        monthly_savings       = total_extra_jobs * CHARGE_PER_JOB
        annual_savings        = monthly_savings * 12
        avg_daily_backlog     = results_df["opt_backlog"].mean()
        peak_backlog_day      = results_df.loc[results_df["opt_backlog"].idxmax(), "day"]
        peak_backlog_val      = results_df["opt_backlog"].max()
        avg_utilization       = np.mean([np.mean(r) for r in results_df["utilization"]])
        days_arr              = results_df["day"].values

        # 30-day total backlog per severity
        sev_backlog_totals     = {sev: sum(r["sev_backlog"][sev]       for _, r in results_df.iterrows()) for sev in SEVERITY_LABELS}
        fibre_sev_bl_totals    = {sev: sum(r["fibre_backlog_sev"][sev]  for _, r in results_df.iterrows()) for sev in SEVERITY_LABELS}
        copper_sev_bl_totals   = {sev: sum(r["copper_backlog_sev"][sev] for _, r in results_df.iterrows()) for sev in SEVERITY_LABELS}

        # ─────────────────────────────────────────────
        # SAVINGS BANNER
        # ─────────────────────────────────────────────
        st.markdown(f"""
        <div class="savings-banner">
            <div style="display:flex; gap:3rem; align-items:center; flex-wrap:wrap;">
                <div>
                    <div style="font-family:'Space Mono',monospace; font-size:0.68rem; text-transform:uppercase;
                         letter-spacing:0.12em; color:#6ee7b7; margin-bottom:0.3rem;">Monthly Cost Savings</div>
                    <div style="font-size:2.4rem; font-weight:700; color:#10b981; line-height:1;">${monthly_savings:,}</div>
                    <div style="font-size:0.8rem; color:#475569; margin-top:0.2rem;">{total_extra_jobs:,} extra jobs × ${CHARGE_PER_JOB}/job</div>
                </div>
                <div style="width:1px; background:rgba(255,255,255,0.1); height:60px;"></div>
                <div>
                    <div style="font-family:'Space Mono',monospace; font-size:0.68rem; text-transform:uppercase;
                         letter-spacing:0.12em; color:#93c5fd; margin-bottom:0.3rem;">Expected Annual Cost Savings</div>
                    <div style="font-size:2.4rem; font-weight:700; color:#60a5fa; line-height:1;">${annual_savings:,}</div>
                    <div style="font-size:0.8rem; color:#475569; margin-top:0.2rem;">Monthly × 12</div>
                </div>
                <div style="width:1px; background:rgba(255,255,255,0.1); height:60px;"></div>
                <div>
                    <div style="font-family:'Space Mono',monospace; font-size:0.68rem; text-transform:uppercase;
                         letter-spacing:0.12em; color:#fbbf24; margin-bottom:0.3rem;">Extra Jobs vs Random</div>
                    <div style="font-size:2.4rem; font-weight:700; color:#f59e0b; line-height:1;">+{total_extra_jobs:,}</div>
                    <div style="font-size:0.8rem; color:#475569; margin-top:0.2rem;">Optimizer vs non-data-driven</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ─────────────────────────────────────────────
        # KPI ROW
        # ─────────────────────────────────────────────
        st.markdown('<div class="section-title">30-Day Summary KPIs</div>', unsafe_allow_html=True)
        k1, k2, k3, k4, k5, k6 = st.columns(6)

        def kpi(col, label, value, sub="", color="", card_color=""):
            col.markdown(f"""
            <div class="kpi-card {card_color}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value {color}">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

        kpi(k1, "Total Jobs (Optimizer)", f"{total_opt_jobs_month:,}",  "30-day total",             color="green", card_color="green")
        kpi(k2, "Total Jobs (Random)",    f"{total_rand_jobs_month:,}", "non-data-driven baseline", color="amber")
        kpi(k3, "Improvement",            f"{total_extra_jobs:+,}",     "extra jobs completed",     color="green")
        kpi(k4, "Avg Daily Backlog",      f"{avg_daily_backlog:.1f}",   "jobs carried fwd/day",     color="amber", card_color="amber")
        kpi(k5, "Peak Backlog",           f"{peak_backlog_val}",        f"on day {peak_backlog_day}",color="red")
        kpi(k6, "Avg Utilization",        f"{avg_utilization:.0%}",     "across all tech types",    color="blue",  card_color="blue")

        st.markdown("<br>", unsafe_allow_html=True)

        # ─────────────────────────────────────────────
        # CHART 1 — OPTIMIZER vs RANDOM
        # ─────────────────────────────────────────────
        st.markdown('<div class="section-title">Daily Performance</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1.6, 1])

        with c1:
            st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
            st.markdown("**Optimizer vs Random — Daily Jobs Completed**")
            fig1, ax1 = plt.subplots(figsize=(9, 4))
            ax1.fill_between(days_arr, results_df["rand_jobs"], results_df["opt_jobs"],
                             alpha=0.25, color="#10b981", label="Extra jobs (gain)")
            ax1.plot(days_arr, results_df["opt_jobs"],  color="#10b981", linewidth=2,   label="Optimizer", zorder=3)
            ax1.plot(days_arr, results_df["rand_jobs"], color="#f59e0b", linewidth=1.5, label="Random",    zorder=3, linestyle="--")
            ax1.set_xlabel("Day", fontsize=9); ax1.set_ylabel("Jobs Completed", fontsize=9)
            ax1.set_title("Daily Throughput Comparison", fontsize=10, color="#94a3b8", pad=10)
            ax1.legend(fontsize=8, framealpha=0.2); ax1.grid(axis="y", zorder=0)
            ax1.spines[["top","right"]].set_visible(False); fig1.tight_layout()
            st.pyplot(fig1)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
            st.markdown("**Daily Total Backlog Carryforward**")
            fig2, ax2 = plt.subplots(figsize=(5, 3.8))
            ax2.bar(days_arr, results_df["opt_backlog"],  color="#ef4444", width=0.6, label="Optimizer backlog", zorder=3, alpha=0.85)
            ax2.bar(days_arr, results_df["rand_backlog"], color="#f59e0b", width=0.6, label="Random backlog",    zorder=3, alpha=0.4)
            ax2.set_xlabel("Day", fontsize=9); ax2.set_ylabel("Backlog Jobs", fontsize=9)
            ax2.set_title("Unmet Jobs → Next Day", fontsize=10, color="#94a3b8", pad=10)
            ax2.legend(fontsize=8, framealpha=0.2); ax2.grid(axis="y", zorder=0)
            ax2.spines[["top","right"]].set_visible(False); fig2.tight_layout()
            st.pyplot(fig2)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ─────────────────────────────────────────────
        # ★ PER-SEVERITY BACKLOG — STACKED DAILY + DONUT
        # ─────────────────────────────────────────────
        st.markdown('<div class="section-title">Backlog by Severity Category</div>', unsafe_allow_html=True)
        bs1, bs2 = st.columns([1.6, 1])

        with bs1:
            st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
            st.markdown("**Daily Backlog Stacked by Severity (Critical → Very Low)**")
            sev_daily = {sev: [r["sev_backlog"][sev] for _, r in results_df.iterrows()] for sev in SEVERITY_LABELS}

            fig3, ax3 = plt.subplots(figsize=(9, 3.8))
            bottoms = np.zeros(len(days_arr))
            for sev in SEVERITY_LABELS:
                vals = np.array(sev_daily[sev])
                ax3.bar(days_arr, vals, bottom=bottoms, width=0.6,
                        label=sev, color=SEV_COLORS[sev], zorder=3, alpha=0.9)
                bottoms += vals
            ax3.set_xlabel("Day", fontsize=9); ax3.set_ylabel("Backlog Jobs", fontsize=9)
            ax3.set_title("Daily Backlog Split by Severity", fontsize=10, color="#94a3b8", pad=10)
            ax3.legend(fontsize=8, framealpha=0.2, loc="upper right")
            ax3.grid(axis="y", zorder=0); ax3.spines[["top","right"]].set_visible(False)
            fig3.tight_layout(); st.pyplot(fig3)
            st.markdown("</div>", unsafe_allow_html=True)

        with bs2:
            st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
            st.markdown("**30-Day Backlog Distribution by Severity**")

            sev_vals = [sev_backlog_totals[sev] for sev in SEVERITY_LABELS]
            if sum(sev_vals) > 0:
                fig4, ax4 = plt.subplots(figsize=(5, 4))
                wedges, texts, autotexts = ax4.pie(
                    sev_vals, labels=SEVERITY_LABELS, labeldistance=1.1,
                    colors=[SEV_COLORS[s] for s in SEVERITY_LABELS],
                    autopct='%1.0f%%', startangle=140, textprops={'fontsize': 9},
                    pctdistance=0.8, wedgeprops=dict(width=0.6, edgecolor="#1a1a2e", linewidth=1.5)
                )
                for at in autotexts: at.set_fontsize(9); at.set_color("white")
                for t in texts: t.set_fontsize(8)
                ax4.set_title("30-Day Backlog Share", fontsize=11, color="#94a3b8", pad=10)
                fig4.tight_layout(); st.pyplot(fig4)
            else:
                st.markdown('<div class="insight-card ok" style="margin-top:1rem;">✅ Zero backlog across all severities!</div>', unsafe_allow_html=True)

            # Per-severity summary table
            st.markdown(f"""
            <table class="styled-table" style="margin-top:0.8rem;">
                <thead><tr><th>Severity</th><th>Total BL</th><th>Avg/Day</th></tr></thead>
                <tbody>
                    {''.join(
                        f'<tr><td><span class="badge badge-critical">{sev}</span></td>'
                        f'<td>{sev_backlog_totals[sev]:,}</td>'
                        f'<td>{sev_backlog_totals[sev]/num_days:.1f}</td></tr>'
                        for idx, sev in enumerate(SEVERITY_LABELS)
                    )}
                </tbody>
            </table>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; color:#94a3b8; padding:2rem;">
    Complete dashboard: Manual + Excel 30-day | Gurobi optimized | KeyError fixed
</div>
""", unsafe_allow_html=True)
