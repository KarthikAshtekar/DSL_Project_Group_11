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
    .badge-ok    { background: rgba(16,185,129,0.15);  color: #10b981; }
    .badge-warn  { background: rgba(245,158,11,0.15);  color: #f59e0b; }
    .badge-error { background: rgba(239,68,68,0.15);   color: #ef4444; }
    .badge-blue  { background: rgba(96,165,250,0.15);  color: #60a5fa; }
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

SEVERITY_COLS_FIBRE  = ["_Critical_Fiber", "_High_fibre"	"_Medium_fibre",	"_Low_fibre", "_Very Low_fibre"]
SEVERITY_COLS_COPPER = ["new_copper_Critical_new",	"new_copper_High_new","new_copper_Medium_new","new_copper_Low_new","new_copper_Very Low_new"]

# Work type index → severity label
SEVERITY_LABELS = ["Critical", "High", "Medium", "Low", "Very Low"]

# ─────────────────────────────────────────────
# TIME MATRIX  (work-type i × tech-type j)
# ─────────────────────────────────────────────
t = {
    (1,1):3.0,(1,2):2.6,(1,3):2.5,(1,4):2.2,(1,5):2.0,
    (2,1):4.0,(2,2):3.4,(2,3):3.25,(2,4):2.8,(2,5):2.5,
    (3,1):5.0,(3,2):4.2,(3,3):4.0,(3,4):3.4,(3,5):3.0,
    (4,1):6.0,(4,2):5.0,(4,3):4.75,(4,4):4.0,(4,5):3.5,
    (5,1):7.0,(5,2):5.8,(5,3):5.5,(5,4):4.6,(5,5):4.0,
}

I = range(1, 6)   # work types (severities)
J = range(1, 6)   # technician types

CHARGE_PER_JOB = 10   # $ per job

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <h1>🔧 30-Day Technician Allocation Optimizer</h1>
  <p>Gurobi-powered · Upload your forecast Excel · See daily optimization, backlog tracking & cost savings vs random assignment</p>
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
        help="Must contain columns: Forecasted_day_index, severity_Critical, severity_High, ..., severity_Very Low_new"
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
    st.caption("Higher β = more priority to New Copper (severity_*_new) jobs")

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
            Expected columns: Forecasted_day_index · severity_Critical · severity_High · severity_Medium · severity_Low · severity_Very Low<br>
            + severity_Critical_new · severity_High_new · severity_Medium_new · severity_Low_new · severity_Very Low_new
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

    # Show preview of loaded data
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

# Fill NaN with 0
for col in SEVERITY_COLS_FIBRE + SEVERITY_COLS_COPPER:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

num_days = len(df)

# ─────────────────────────────────────────────
# DAY-BY-DAY OPTIMIZATION WITH BACKLOG
# ─────────────────────────────────────────────
with st.spinner(f"Running Gurobi optimizer across {num_days} days..."):

    # Results storage
    daily_results = []

    # Backlog carries forward: dict {work_type_i: backlog_count}
    # Fibre backlog and Copper backlog tracked separately
    fibre_backlog  = {i: 0 for i in I}   # i=1..5 → severity index
    copper_backlog = {i: 0 for i in I}

    # For random baseline: naive round-robin / proportional assignment
    # Random = assign min(demand, floor(capacity / num_work_types)) per type
    random_fibre_backlog  = {i: 0 for i in I}
    random_copper_backlog = {i: 0 for i in I}

    for day_idx, row in df.iterrows():
        day_num = int(row["Forecasted_day_index"])

        # ── Today's demand (forecast + yesterday's backlog) ──────────
        F_demand = {}   # fibre demand per work type i
        C_demand = {}   # copper demand per work type i

        for i in I:
            sev_label = SEVERITY_COLS_FIBRE[i-1]
            sev_label_c = SEVERITY_COLS_COPPER[i-1]
            F_demand[i] = int(row[sev_label]) + fibre_backlog[i]
            C_demand[i] = int(row[sev_label_c]) + copper_backlog[i]

        # Backlog demand (all combined into a Backlog work type proportionally)
        # We use fibre as "Fibre" type, copper as "Copper" type, no separate backlog category
        # Backlog is already embedded in F_demand and C_demand

        # ── GUROBI MODEL ────────────────────────────────────────────
        model = gp.Model(f"Day_{day_num}")
        model.setParam("OutputFlag", 0)

        xF = model.addVars(I, J, vtype=GRB.INTEGER, lb=0, name="Fibre")
        xC = model.addVars(I, J, vtype=GRB.INTEGER, lb=0, name="Copper")

        model.setObjective(
            gp.quicksum(xF[i,j] for i in I for j in J)
            + beta * gp.quicksum(xC[i,j] for i in I for j in J),
            GRB.MAXIMIZE
        )

        # Demand constraints
        for i in I:
            model.addConstr(gp.quicksum(xF[i,j] for j in J) <= F_demand[i], f"FibreDemand_{i}")
            model.addConstr(gp.quicksum(xC[i,j] for j in J) <= C_demand[i], f"CopperDemand_{i}")

        # Capacity constraints
        for j in J:
            model.addConstr(
                gp.quicksum(t[i,j] * (xF[i,j] + xC[i,j]) for i in I) <= workers[j] * H,
                f"Capacity_{j}"
            )
            # Copper 2-day window
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

        # ── BACKLOG UPDATE ───────────────────────────────────────────
        new_fibre_backlog  = {}
        new_copper_backlog = {}
        for i in I:
            new_fibre_backlog[i]  = max(0, F_demand[i] - opt_fibre[i])
            new_copper_backlog[i] = max(0, C_demand[i] - opt_copper[i])

        total_backlog_today = sum(new_fibre_backlog.values()) + sum(new_copper_backlog.values())

        # ── RANDOM BASELINE ──────────────────────────────────────────
        # Random: distribute capacity evenly across work types, ignore severity
        total_capacity_hours = sum(workers[j] * H for j in J)
        # Average job time (simple mean across all i,j)
        avg_job_time = np.mean(list(t.values()))
        rand_total_capacity_jobs = int(total_capacity_hours / avg_job_time)

        # Random also has its own backlog
        rand_F_demand = {}
        rand_C_demand = {}
        for i in I:
            rand_F_demand[i] = int(row[SEVERITY_COLS_FIBRE[i-1]]) + random_fibre_backlog[i]
            rand_C_demand[i] = int(row[SEVERITY_COLS_COPPER[i-1]]) + random_copper_backlog[i]

        rand_total_demand = sum(rand_F_demand.values()) + sum(rand_C_demand.values())
        rand_jobs = min(rand_total_capacity_jobs, rand_total_demand)

        # Random backlog
        rand_unmet = max(0, rand_total_demand - rand_jobs)
        # Distribute rand backlog proportionally
        if rand_total_demand > 0:
            for i in I:
                share = (rand_F_demand[i] + rand_C_demand[i]) / rand_total_demand
                rand_unmet_i = int(rand_unmet * share)
                random_fibre_backlog[i]  = int(rand_unmet_i * rand_F_demand[i] / (rand_F_demand[i] + rand_C_demand[i] + 1e-9))
                random_copper_backlog[i] = rand_unmet_i - random_fibre_backlog[i]
        else:
            random_fibre_backlog  = {i: 0 for i in I}
            random_copper_backlog = {i: 0 for i in I}

        rand_total_backlog = sum(random_fibre_backlog.values()) + sum(random_copper_backlog.values())

        # Store results
        daily_results.append({
            "day":                  day_num,
            "fibre_demand_fresh":   sum(int(row[SEVERITY_COLS_FIBRE[i-1]]) for i in I),
            "copper_demand_fresh":  sum(int(row[SEVERITY_COLS_COPPER[i-1]]) for i in I),
            "fibre_demand_total":   sum(F_demand.values()),
            "copper_demand_total":  sum(C_demand.values()),
            "opt_fibre":            total_opt_fibre,
            "opt_copper":           total_opt_copper,
            "opt_jobs":             total_opt_jobs,
            "opt_backlog":          total_backlog_today,
            "opt_fibre_backlog":    dict(new_fibre_backlog),
            "opt_copper_backlog":   dict(new_copper_backlog),
            "rand_jobs":            rand_jobs,
            "rand_backlog":         rand_total_backlog,
            "extra_jobs":           max(0, total_opt_jobs - rand_jobs),
            "utilization":          [
                sum(t[i,j]*(opt_fibre.get(i,0)+opt_copper.get(i,0)) for i in I) / (workers[j]*H)
                if workers[j]*H > 0 else 0
                for j in J
            ],
        })

        # Update backlog for next day
        fibre_backlog  = new_fibre_backlog
        copper_backlog = new_copper_backlog

results_df = pd.DataFrame(daily_results)

# ─────────────────────────────────────────────
# AGGREGATE METRICS
# ─────────────────────────────────────────────
total_opt_jobs_month   = results_df["opt_jobs"].sum()
total_rand_jobs_month  = results_df["rand_jobs"].sum()
total_extra_jobs       = results_df["extra_jobs"].sum()
monthly_savings        = total_extra_jobs * CHARGE_PER_JOB
annual_savings         = monthly_savings * 12
avg_daily_backlog      = results_df["opt_backlog"].mean()
peak_backlog_day       = results_df.loc[results_df["opt_backlog"].idxmax(), "day"]
peak_backlog_val       = results_df["opt_backlog"].max()
avg_utilization        = np.mean([np.mean(r) for r in results_df["utilization"]])

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

kpi(k1, "Total Jobs (Optimizer)", f"{total_opt_jobs_month:,}",  "30-day total",            color="green", card_color="green")
kpi(k2, "Total Jobs (Random)",    f"{total_rand_jobs_month:,}", "non-data-driven baseline", color="amber")
kpi(k3, "Improvement",            f"{total_extra_jobs:+,}",     "extra jobs completed",     color="green")
kpi(k4, "Avg Daily Backlog",      f"{avg_daily_backlog:.1f}",   "jobs carried fwd/day",     color="amber", card_color="amber")
kpi(k5, "Peak Backlog",           f"{peak_backlog_val}",        f"on day {peak_backlog_day}",color="red")
kpi(k6, "Avg Utilization",        f"{avg_utilization:.0%}",     "across all tech types",    color="blue",  card_color="blue")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHART 1 — DAILY JOBS: OPTIMIZER vs RANDOM
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Daily Performance</div>', unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns([1.6, 1])

with chart_col1:
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.markdown("**Optimizer vs Random — Daily Jobs Completed**")
    fig1, ax1 = plt.subplots(figsize=(9, 3.8))
    days = results_df["day"].values
    ax1.fill_between(days, results_df["rand_jobs"], results_df["opt_jobs"],
                     alpha=0.25, color="#10b981", label="Extra jobs (gain)")
    ax1.plot(days, results_df["opt_jobs"],  color="#10b981", linewidth=2,   label="Optimizer", zorder=3)
    ax1.plot(days, results_df["rand_jobs"], color="#f59e0b", linewidth=1.5, label="Random",    zorder=3, linestyle="--")
    ax1.set_xlabel("Day", fontsize=9)
    ax1.set_ylabel("Jobs Completed", fontsize=9)
    ax1.set_title("Daily Throughput Comparison", fontsize=10, color="#94a3b8", pad=10)
    ax1.legend(fontsize=8, framealpha=0.2)
    ax1.grid(axis="y", zorder=0)
    ax1.spines[["top","right"]].set_visible(False)
    fig1.tight_layout()
    st.pyplot(fig1)
    st.markdown("</div>", unsafe_allow_html=True)

with chart_col2:
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.markdown("**Daily Backlog Carryforward**")
    fig2, ax2 = plt.subplots(figsize=(5, 3.8))
    ax2.bar(days, results_df["opt_backlog"],  color="#ef4444", width=0.6, label="Optimizer backlog", zorder=3, alpha=0.85)
    ax2.bar(days, results_df["rand_backlog"], color="#f59e0b", width=0.6, label="Random backlog",    zorder=3, alpha=0.4)
    ax2.set_xlabel("Day", fontsize=9)
    ax2.set_ylabel("Backlog Jobs", fontsize=9)
    ax2.set_title("Unmet Jobs → Next Day", fontsize=10, color="#94a3b8", pad=10)
    ax2.legend(fontsize=8, framealpha=0.2)
    ax2.grid(axis="y", zorder=0)
    ax2.spines[["top","right"]].set_visible(False)
    fig2.tight_layout()
    st.pyplot(fig2)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHART 2 — DEMAND vs COMPLETED (STACKED)
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Demand & Fulfillment</div>', unsafe_allow_html=True)

dem_col1, dem_col2 = st.columns([1.6, 1])

with dem_col1:
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.markdown("**Daily Demand (Fresh + Backlog) vs Completed**")
    fig3, ax3 = plt.subplots(figsize=(9, 3.8))
    total_demand_daily = results_df["fibre_demand_total"] + results_df["copper_demand_total"]
    ax3.bar(days, results_df["opt_fibre"],  color="#60a5fa", width=0.6, label="Fibre completed",  zorder=3)
    ax3.bar(days, results_df["opt_copper"], color="#34d399", width=0.6, label="Copper completed",
            bottom=results_df["opt_fibre"], zorder=3)
    ax3.plot(days, total_demand_daily, color="#f472b6", linewidth=1.5, linestyle="--",
             label="Total demand (incl. backlog)", zorder=4)
    ax3.set_xlabel("Day", fontsize=9)
    ax3.set_ylabel("Jobs", fontsize=9)
    ax3.set_title("Fibre & Copper Completion vs Total Demand", fontsize=10, color="#94a3b8", pad=10)
    ax3.legend(fontsize=8, framealpha=0.2)
    ax3.grid(axis="y", zorder=0)
    ax3.spines[["top","right"]].set_visible(False)
    fig3.tight_layout()
    st.pyplot(fig3)
    st.markdown("</div>", unsafe_allow_html=True)

with dem_col2:
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.markdown("**Monthly Fibre vs Copper Split**")
    fig4, ax4 = plt.subplots(figsize=(4.5, 3.8))
    wedge_colors = ["#60a5fa", "#34d399"]
    vals = [results_df["opt_fibre"].sum(), results_df["opt_copper"].sum()]
    wedges, texts, autotexts = ax4.pie(
        vals, labels=["Fibre", "Copper (New)"],
        colors=wedge_colors, autopct="%1.0f%%", startangle=140,
        wedgeprops=dict(width=0.65, edgecolor="#1a1a2e", linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(9); at.set_color("white")
    ax4.set_title("Job Mix (30 days)", fontsize=10, color="#94a3b8", pad=12)
    fig4.tight_layout()
    st.pyplot(fig4)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHART 3 — COST SAVINGS CUMULATIVE
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Cost Savings Analysis</div>', unsafe_allow_html=True)

sav_col1, sav_col2 = st.columns([1.6, 1])

with sav_col1:
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.markdown("**Cumulative Cost Savings vs Random Assignment**")
    fig5, ax5 = plt.subplots(figsize=(9, 3.5))
    cumulative_savings = (results_df["extra_jobs"] * CHARGE_PER_JOB).cumsum()
    ax5.fill_between(days, cumulative_savings, alpha=0.2, color="#10b981")
    ax5.plot(days, cumulative_savings, color="#10b981", linewidth=2.5, zorder=3)
    # Annotate final value
    ax5.annotate(f"${cumulative_savings.iloc[-1]:,.0f}",
                 xy=(days[-1], cumulative_savings.iloc[-1]),
                 xytext=(-40, 12), textcoords="offset points",
                 fontsize=10, color="#10b981", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#10b981", lw=1.2))
    ax5.set_xlabel("Day", fontsize=9)
    ax5.set_ylabel("Cumulative Savings ($)", fontsize=9)
    ax5.set_title("Running Total Savings (Optimizer vs Random)", fontsize=10, color="#94a3b8", pad=10)
    ax5.grid(axis="y", zorder=0)
    ax5.spines[["top","right"]].set_visible(False)
    fig5.tight_layout()
    st.pyplot(fig5)
    st.markdown("</div>", unsafe_allow_html=True)

with sav_col2:
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
# BACKLOG AGE ANALYSIS
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Backlog Age & Origin Tracking</div>', unsafe_allow_html=True)

bl_col1, bl_col2 = st.columns([1.6, 1])

with bl_col1:
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.markdown("**Backlog Carryover — How Many Days Old?**")

    # Reconstruct backlog age: for each day show how many backlog jobs are from d-1, d-2, d-3+
    # We approximate from the rolling backlog series
    backlog_series = results_df["opt_backlog"].values

    # Age buckets: d-1 (yesterday's), d-2, d-3+
    age_d1 = []
    age_d2 = []
    age_d3 = []

    for d in range(len(backlog_series)):
        bl_today = backlog_series[d]
        bl_prev1 = backlog_series[d-1] if d >= 1 else 0
        bl_prev2 = backlog_series[d-2] if d >= 2 else 0

        # Fresh backlog created today (not from previous)
        new_bl  = max(0, bl_today - bl_prev1)
        from_d1 = min(bl_prev1, bl_today - new_bl)
        from_d2 = min(bl_prev2, max(0, bl_today - new_bl - from_d1))
        from_d3 = max(0, bl_today - new_bl - from_d1 - from_d2)

        age_d1.append(new_bl)
        age_d2.append(from_d1)
        age_d3.append(from_d2 + from_d3)

    fig6, ax6 = plt.subplots(figsize=(9, 3.5))
    ax6.bar(days, age_d1, color="#f59e0b", width=0.6, label="New today (day 0)",  zorder=3)
    ax6.bar(days, age_d2, color="#ef4444", width=0.6, label="From day -1",
            bottom=age_d1, zorder=3)
    ax6.bar(days, age_d3, color="#7c3aed", width=0.6, label="From day -2+",
            bottom=[a+b for a,b in zip(age_d1, age_d2)], zorder=3)
    ax6.set_xlabel("Day", fontsize=9)
    ax6.set_ylabel("Backlog Jobs", fontsize=9)
    ax6.set_title("Backlog Composition by Age", fontsize=10, color="#94a3b8", pad=10)
    ax6.legend(fontsize=8, framealpha=0.2)
    ax6.grid(axis="y", zorder=0)
    ax6.spines[["top","right"]].set_visible(False)
    fig6.tight_layout()
    st.pyplot(fig6)
    st.markdown("</div>", unsafe_allow_html=True)

with bl_col2:
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.markdown("**🧠 Backlog Insights**")

    total_backlog_jobs = results_df["opt_backlog"].sum()
    days_with_backlog  = (results_df["opt_backlog"] > 0).sum()
    max_age_days       = max(
        (i for i, v in enumerate(backlog_series) if v > 0),
        default=0
    )

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

    if total_extra_jobs > 0:
        insights.append(("ok", f"💰 Optimizer adds ${monthly_savings:,}/month vs random"))
    if avg_utilization > 0.9:
        insights.append(("warn", "⚠️ Avg utilization >90% — near capacity limit"))
    elif avg_utilization < 0.5:
        insights.append(("info", f"💡 Avg utilization {avg_utilization:.0%} — capacity headroom available"))

    for kind, msg in insights:
        st.markdown(f'<div class="insight-card {kind}">{msg}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DAILY DETAIL TABLE
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Daily Detail Table</div>', unsafe_allow_html=True)

with st.expander("📋 Show full 30-day breakdown", expanded=False):
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
            <td>{util_badge}</td>
            <td style="color:#10b981">${daily_saving}</td>
        </tr>""")

    st.markdown(f"""
    <div class="chart-panel">
    <table class="styled-table">
        <thead><tr>
            <th>Day</th>
            <th>Fibre (Fresh)</th>
            <th>Copper (Fresh)</th>
            <th>Total Demand</th>
            <th>Opt Jobs</th>
            <th>Rand Jobs</th>
            <th>Extra Jobs</th>
            <th>Backlog</th>
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
    GUROBI OPTIMIZER · 30-DAY FORECAST · BACKLOG TRACKING · COST SAVINGS ANALYSIS
</div>
""", unsafe_allow_html=True)
