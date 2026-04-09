import streamlit as st
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="30-Day Technician Optimizer",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e", "axes.facecolor":  "#1a1a2e",
    "axes.edgecolor":   "#2d2d4e", "axes.labelcolor": "#94a3b8",
    "xtick.color":      "#64748b", "ytick.color":     "#64748b",
    "text.color":       "#e0e0f0", "grid.color":      "#2d2d4e",
    "font.family":      "monospace",
})

# ─────────────────────────────────────────────
# COLUMN MAPPING — matches your Excel exactly
# ─────────────────────────────────────────────
SEVERITY_COLS_FIBRE  = ["_Critical_Fiber", "_High_fibre", "_Medium_fibre", "_Low_fibre", "_Very Low_fibre"]
SEVERITY_COLS_COPPER = ["new_copper_Critical", "new_copper_High", "new_copper_Medium", "new_copper_Low", "new_copper_Very Low"]

SEVERITY_LABELS = ["Critical", "High", "Medium", "Low", "Very Low"]

SEV_COLORS = {
    "Critical": "#ef4444",
    "High":     "#f59e0b",
    "Medium":   "#a78bfa",
    "Low":      "#60a5fa",
    "Very Low": "#34d399",
}
SEV_BADGE_CLS = ["badge-critical", "badge-high", "badge-medium", "badge-low", "badge-vlow"]

# ─────────────────────────────────────────────
# TIME MATRIX
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <h1>🔧 30-Day Technician Allocation Optimizer</h1>
  <p>Gurobi-powered · Upload your forecast Excel · Daily optimization · Per-severity backlog tracking · Cost savings vs random assignment</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Forecast Data")
    uploaded_file = st.file_uploader(
        "Upload 30-day forecast Excel",
        type=["xlsx", "xls"],
        help="Columns: Forecasted_day_index, _Critical_Fiber, _High_fibre, _Medium_fibre, _Low_fibre, _Very Low_fibre, new_copper_Critical, new_copper_High, new_copper_Medium, new_copper_Low, new_copper_Very Low"
    )

    st.markdown("### ⚙️ Configuration")
    H = st.number_input("Work hours per day", min_value=1, max_value=24, value=8)

    st.markdown("### 👷 Technicians Available")
    workers = {
        j: st.number_input(f"Type {j}", min_value=0, value=5, key=f"w{j}")
        for j in range(1, 6)
    }

    st.markdown("### 🎛️ Priority Weight")
    beta = st.slider("New Copper Weight (β)", 0.0, 1.0, 0.5, 0.05)
    st.caption("Higher β = more priority to New Copper jobs")

    st.markdown("---")
    run = st.button("🚀 Run 30-Day Optimization")

# ─────────────────────────────────────────────
# IDLE / NO FILE STATE
# ─────────────────────────────────────────────
if uploaded_file is None:
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; color:#334155;">
        <div style="font-size:4rem; margin-bottom:1rem;">📊</div>
        <div style="font-family:'Space Mono',monospace; font-size:1rem; color:#475569; margin-bottom:0.5rem;">
            Upload your 30-day forecast Excel in the sidebar
        </div>
        <div style="font-size:0.82rem; color:#334155;">
            Expected columns: Forecasted_day_index · _Critical_Fiber · _High_fibre · _Medium_fibre · _Low_fibre · _Very Low_fibre<br>
            + new_copper_Critical · new_copper_High · new_copper_Medium · new_copper_Low · new_copper_Very Low
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not run:
    st.markdown("""
    <div style="text-align:center; padding:3rem 2rem; color:#334155;">
        <div style="font-size:4rem; margin-bottom:1rem;">⚡</div>
        <div style="font-family:'Space Mono',monospace; font-size:1rem; color:#475569;">
            File loaded! Set technician counts and click Run Optimization
        </div>
    </div>
    """, unsafe_allow_html=True)
    try:
        df_preview = pd.read_excel(uploaded_file, nrows=5)
        st.markdown('<div class="section-title">Data Preview (first 5 rows)</div>', unsafe_allow_html=True)
        st.dataframe(df_preview, use_container_width=True)
    except Exception as e:
        st.error(f"Could not read Excel file: {e}")
    st.stop()

# ─────────────────────────────────────────────
# LOAD & VALIDATE EXCEL
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# DAY-BY-DAY GUROBI OPTIMIZATION WITH BACKLOG
# ─────────────────────────────────────────────
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
    fig1, ax1 = plt.subplots(figsize=(9, 3.8))
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
        fig4, ax4 = plt.subplots(figsize=(4.5, 3.2))
        wedges, texts, autotexts = ax4.pie(
            sev_vals, labels=SEVERITY_LABELS,
            colors=[SEV_COLORS[s] for s in SEVERITY_LABELS],
            autopct="%1.0f%%", startangle=140,
            wedgeprops=dict(width=0.65, edgecolor="#1a1a2e", linewidth=2)
        )
        for at in autotexts: at.set_fontsize(8); at.set_color("white")
        ax4.set_title("Backlog Share (30 days)", fontsize=10, color="#94a3b8", pad=8)
        fig4.tight_layout(); st.pyplot(fig4)
    else:
        st.markdown('<div class="insight-card ok" style="margin-top:1rem;">✅ Zero backlog across all severities!</div>', unsafe_allow_html=True)

    # Per-severity summary table
    st.markdown(f"""
    <table class="styled-table" style="margin-top:0.8rem;">
        <thead><tr><th>Severity</th><th>Total BL</th><th>Avg/Day</th></tr></thead>
        <tbody>
            {''.join(
                f'<tr><td><span class="badge {SEV_BADGE_CLS[idx]}">{sev}</span></td>'
                f'<td>{sev_backlog_totals[sev]:,}</td>'
                f'<td>{sev_backlog_totals[sev]/num_days:.1f}</td></tr>'
                for idx, sev in enumerate(SEVERITY_LABELS)
            )}
        </tbody>
    </table>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ★ PER-SEVERITY FIBRE vs COPPER BACKLOG SPLIT
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Severity Backlog — Fibre vs Copper Split</div>', unsafe_allow_html=True)
st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
st.markdown("**Accumulated 30-Day Backlog per Severity — Fibre vs Copper**")

fig5, ax5 = plt.subplots(figsize=(12, 3.5))
x_sev = np.arange(5)
bar_w = 0.35
b1 = ax5.bar(x_sev - bar_w/2, [fibre_sev_bl_totals[s]  for s in SEVERITY_LABELS],
             bar_w, label="Fibre backlog",  color="#60a5fa", zorder=3)
b2 = ax5.bar(x_sev + bar_w/2, [copper_sev_bl_totals[s] for s in SEVERITY_LABELS],
             bar_w, label="Copper backlog", color="#34d399", zorder=3)
ax5.set_xticks(x_sev); ax5.set_xticklabels(SEVERITY_LABELS, fontsize=9)
ax5.set_ylabel("Total Backlog Jobs (30 days)", fontsize=9)
ax5.set_title("Fibre vs Copper Backlog per Severity Category", fontsize=10, color="#94a3b8", pad=10)
ax5.legend(fontsize=8, framealpha=0.2); ax5.grid(axis="y", zorder=0)
ax5.spines[["top","right"]].set_visible(False)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    if h > 0:
        ax5.text(bar.get_x() + bar.get_width()/2, h + 0.3, str(int(h)),
                 ha="center", va="bottom", fontsize=8, color="white")
fig5.tight_layout(); st.pyplot(fig5)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DEMAND vs COMPLETED + JOB MIX
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Demand & Fulfillment</div>', unsafe_allow_html=True)
dc1, dc2 = st.columns([1.6, 1])

with dc1:
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.markdown("**Daily Demand (Fresh + Backlog) vs Completed**")
    fig6, ax6 = plt.subplots(figsize=(9, 3.8))
    total_demand_daily = results_df["fibre_demand_total"] + results_df["copper_demand_total"]
    ax6.bar(days_arr, results_df["opt_fibre"],  color="#60a5fa", width=0.6, label="Fibre completed",  zorder=3)
    ax6.bar(days_arr, results_df["opt_copper"], color="#34d399", width=0.6, label="Copper completed",
            bottom=results_df["opt_fibre"], zorder=3)
    ax6.plot(days_arr, total_demand_daily, color="#f472b6", linewidth=1.5, linestyle="--",
             label="Total demand (incl. backlog)", zorder=4)
    ax6.set_xlabel("Day", fontsize=9); ax6.set_ylabel("Jobs", fontsize=9)
    ax6.set_title("Fibre & Copper Completion vs Total Demand", fontsize=10, color="#94a3b8", pad=10)
    ax6.legend(fontsize=8, framealpha=0.2); ax6.grid(axis="y", zorder=0)
    ax6.spines[["top","right"]].set_visible(False); fig6.tight_layout()
    st.pyplot(fig6)
    st.markdown("</div>", unsafe_allow_html=True)

with dc2:
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.markdown("**Monthly Fibre vs Copper Split**")
    fig7, ax7 = plt.subplots(figsize=(4.5, 3.8))
    vals = [results_df["opt_fibre"].sum(), results_df["opt_copper"].sum()]
    wedges, texts, autotexts = ax7.pie(
        vals, labels=["Fibre", "Copper (New)"],
        colors=["#60a5fa", "#34d399"], autopct="%1.0f%%", startangle=140,
        wedgeprops=dict(width=0.65, edgecolor="#1a1a2e", linewidth=2)
    )
    for at in autotexts: at.set_fontsize(9); at.set_color("white")
    ax7.set_title("Job Mix (30 days)", fontsize=10, color="#94a3b8", pad=12)
    fig7.tight_layout(); st.pyplot(fig7)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# COST SAVINGS
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Cost Savings Analysis</div>', unsafe_allow_html=True)
sv1, sv2 = st.columns([1.6, 1])

with sv1:
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.markdown("**Cumulative Cost Savings vs Random Assignment**")
    fig8, ax8 = plt.subplots(figsize=(9, 3.5))
    cumulative_savings = (results_df["extra_jobs"] * CHARGE_PER_JOB).cumsum()
    ax8.fill_between(days_arr, cumulative_savings, alpha=0.2, color="#10b981")
    ax8.plot(days_arr, cumulative_savings, color="#10b981", linewidth=2.5, zorder=3)
    ax8.annotate(f"${cumulative_savings.iloc[-1]:,.0f}",
                 xy=(days_arr[-1], cumulative_savings.iloc[-1]),
                 xytext=(-40, 12), textcoords="offset points",
                 fontsize=10, color="#10b981", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#10b981", lw=1.2))
    ax8.set_xlabel("Day", fontsize=9); ax8.set_ylabel("Cumulative Savings ($)", fontsize=9)
    ax8.set_title("Running Total Savings (Optimizer vs Random)", fontsize=10, color="#94a3b8", pad=10)
    ax8.grid(axis="y", zorder=0); ax8.spines[["top","right"]].set_visible(False)
    fig8.tight_layout(); st.pyplot(fig8)
    st.markdown("</div>", unsafe_allow_html=True)

with sv2:
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.markdown("**Savings Breakdown**")
    st.markdown(f"""
    <table class="styled-table">
        <thead><tr><th>Metric</th><th>Value</th></tr></thead>
        <tbody>
            <tr><td>Extra jobs / month</td><td><b>+{total_extra_jobs:,}</b></td></tr>
            <tr><td>Charge per job</td><td>${CHARGE_PER_JOB}</td></tr>
            <tr><td>Monthly savings</td><td style="color:#10b981"><b>${monthly_savings:,}</b></td></tr>
            <tr><td>Expected Annual savings (×12)</td><td style="color:#60a5fa"><b>${annual_savings:,}</b></td></tr>
            <tr><td>Optimizer jobs</td><td>{total_opt_jobs_month:,}</td></tr>
            <tr><td>Random jobs</td><td>{total_rand_jobs_month:,}</td></tr>
            <tr><td>Improvement %</td><td style="color:#10b981"><b>{(total_extra_jobs/max(total_rand_jobs_month,1)*100):.1f}%</b></td></tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# BACKLOG AGE
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Backlog Age & Origin Tracking</div>', unsafe_allow_html=True)
ba1, ba2 = st.columns([1.6, 1])

with ba1:
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.markdown("**Backlog Carryover — How Many Days Old?**")
    backlog_series = results_df["opt_backlog"].values
    age_d1, age_d2, age_d3 = [], [], []
    for d in range(len(backlog_series)):
        bl_today = backlog_series[d]
        bl_prev1 = backlog_series[d-1] if d >= 1 else 0
        bl_prev2 = backlog_series[d-2] if d >= 2 else 0
        new_bl  = max(0, bl_today - bl_prev1)
        from_d1 = min(bl_prev1, bl_today - new_bl)
        from_d2 = min(bl_prev2, max(0, bl_today - new_bl - from_d1))
        from_d3 = max(0, bl_today - new_bl - from_d1 - from_d2)
        age_d1.append(new_bl); age_d2.append(from_d1); age_d3.append(from_d2 + from_d3)

    fig9, ax9 = plt.subplots(figsize=(9, 3.5))
    ax9.bar(days_arr, age_d1, color="#f59e0b", width=0.6, label="New today (day 0)",  zorder=3)
    ax9.bar(days_arr, age_d2, color="#ef4444", width=0.6, label="From day -1",
            bottom=age_d1, zorder=3)
    ax9.bar(days_arr, age_d3, color="#7c3aed", width=0.6, label="From day -2+",
            bottom=[a+b for a,b in zip(age_d1, age_d2)], zorder=3)
    ax9.set_xlabel("Day", fontsize=9); ax9.set_ylabel("Backlog Jobs", fontsize=9)
    ax9.set_title("Backlog Composition by Age", fontsize=10, color="#94a3b8", pad=10)
    ax9.legend(fontsize=8, framealpha=0.2); ax9.grid(axis="y", zorder=0)
    ax9.spines[["top","right"]].set_visible(False); fig9.tight_layout()
    st.pyplot(fig9)
    st.markdown("</div>", unsafe_allow_html=True)

with ba2:
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.markdown("**🧠 Backlog Insights**")
    days_with_backlog = (results_df["opt_backlog"] > 0).sum()
    insights = []
    if avg_daily_backlog > 10:
        insights.append(("warn", f"⚠️ High avg daily backlog: {avg_daily_backlog:.1f} jobs/day"))
    if peak_backlog_val > 0:
        insights.append(("error" if peak_backlog_val > 20 else "warn",
                         f"📌 Peak backlog {peak_backlog_val} jobs on day {peak_backlog_day}"))
    if days_with_backlog == num_days:
        insights.append(("error", "🚨 Backlog every single day — consider more technicians"))
    elif days_with_backlog > num_days * 0.7:
        insights.append(("warn", f"⚠️ Backlog on {days_with_backlog}/{num_days} days"))
    else:
        insights.append(("ok", f"✅ Backlog-free on {num_days - days_with_backlog} days"))
    # Highest-severity backlog alert
    if sev_backlog_totals["Critical"] > 0:
        insights.append(("error", f"🚨 {sev_backlog_totals['Critical']} Critical jobs in backlog over 30 days"))
    if sev_backlog_totals["High"] > 0:
        insights.append(("warn",  f"⚠️ {sev_backlog_totals['High']} High-severity jobs in backlog"))
    if total_extra_jobs > 0:
        insights.append(("ok", f"💰 Optimizer adds ${monthly_savings:,}/month vs random"))
    if avg_utilization > 0.9:
        insights.append(("warn", "⚠️ Avg utilization >90% — near capacity limit"))
    elif avg_utilization < 0.5:
        insights.append(("info", f"💡 Avg utilization {avg_utilization:.0%} — headroom available"))
    for kind, msg in insights:
        st.markdown(f'<div class="insight-card {kind}">{msg}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DAILY DETAIL TABLE — with per-severity backlog columns
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Daily Detail Table</div>', unsafe_allow_html=True)

with st.expander("📋 Show full 30-day breakdown (incl. per-severity backlog)", expanded=False):
    table_rows = []
    for _, r in results_df.iterrows():
        backlog = r["opt_backlog"]
        bl_badge = (
            f'<span class="badge badge-error">{backlog}</span>' if backlog > 20 else
            f'<span class="badge badge-warn">{backlog}</span>'  if backlog > 5  else
            f'<span class="badge badge-ok">{backlog}</span>'
        )
        extra = r["extra_jobs"]
        extra_badge = (
            f'<span class="badge badge-ok">+{extra}</span>' if extra > 0 else
            f'<span class="badge badge-warn">0</span>'
        )
        total_demand = r["fibre_demand_total"] + r["copper_demand_total"]
        avg_util = np.mean(r["utilization"])
        util_badge = (
            f'<span class="badge badge-error">{avg_util:.0%}</span>' if avg_util > 0.9 else
            f'<span class="badge badge-warn">{avg_util:.0%}</span>'  if avg_util > 0.7 else
            f'<span class="badge badge-ok">{avg_util:.0%}</span>'
        )
        daily_saving = extra * CHARGE_PER_JOB

        # Per-severity backlog cells with coloured badges
        sev_cells = "".join(
            f'<td><span class="badge {SEV_BADGE_CLS[idx]}">{r["sev_backlog"][sev]}</span></td>'
            for idx, sev in enumerate(SEVERITY_LABELS)
        )

        table_rows.append(f"""
        <tr>
            <td><b>Day {int(r['day'])}</b></td>
            <td>{int(r['fibre_demand_fresh'])}</td>
            <td>{int(r['copper_demand_fresh'])}</td>
            <td>{int(total_demand)}</td>
            <td>{int(r['opt_jobs'])}</td>
            <td>{int(r['rand_jobs'])}</td>
            <td>{extra_badge}</td>
            <td>{bl_badge}</td>
            {sev_cells}
            <td>{util_badge}</td>
            <td style="color:#10b981">${daily_saving}</td>
        </tr>""")

    sev_header_cells = "".join(f'<th>BL {sev}</th>' for sev in SEVERITY_LABELS)

    st.markdown(f"""
    <div class="chart-panel" style="overflow-x:auto;">
    <table class="styled-table">
        <thead><tr>
            <th>Day</th>
            <th>Fibre Fresh</th>
            <th>Copper Fresh</th>
            <th>Total Demand</th>
            <th>Opt Jobs</th>
            <th>Rand Jobs</th>
            <th>Extra Jobs</th>
            <th>Total Backlog</th>
            {sev_header_cells}
            <th>Avg Util</th>
            <th>Daily Saving</th>
        </tr></thead>
        <tbody>{''.join(table_rows)}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#334155; font-family:'Space Mono',monospace; font-size:0.7rem; padding:1rem;">
    GUROBI OPTIMIZER · 30-DAY FORECAST · PER-SEVERITY BACKLOG TRACKING · COST SAVINGS ANALYSIS
</div>
""", unsafe_allow_html=True)
