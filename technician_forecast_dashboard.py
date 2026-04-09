import streamlit as st
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="30-Day Technician Optimizer",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.stApp {
    background: linear-gradient(160deg, #080c18 0%, #0d1424 60%, #0a1020 100%);
    color: #c8d4e8;
}

[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.03);
    border-right: 1px solid rgba(255,255,255,0.07);
}

/* Header */
.dash-header {
    background: linear-gradient(90deg, rgba(56,189,248,0.12), rgba(99,102,241,0.07));
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 16px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.dash-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: #f0f6ff;
    margin: 0;
    letter-spacing: -0.03em;
}
.dash-header p { color: #64748b; margin: 0.3rem 0 0; font-size: 0.85rem; }

/* KPI cards */
.kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.75rem; margin-bottom: 1.25rem; }
.kpi-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.kpi-card:hover { border-color: rgba(56,189,248,0.3); }
.kpi-label { font-size: 0.65rem; font-family: 'IBM Plex Mono', monospace; text-transform: uppercase; letter-spacing: 0.12em; color: #475569; margin-bottom: 0.35rem; }
.kpi-value { font-size: 1.9rem; font-weight: 600; line-height: 1; }
.kpi-sub { font-size: 0.72rem; color: #475569; margin-top: 0.3rem; }
.kpi-blue  { color: #38bdf8; }
.kpi-green { color: #34d399; }
.kpi-amber { color: #fbbf24; }
.kpi-red   { color: #f87171; }
.kpi-purple{ color: #a78bfa; }

/* Section title */
.sec-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #334155;
    margin-bottom: 0.85rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    padding-bottom: 0.45rem;
}

/* Panel */
.panel {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

/* Savings highlight */
.savings-box {
    background: linear-gradient(135deg, rgba(52,211,153,0.12), rgba(56,189,248,0.06));
    border: 1px solid rgba(52,211,153,0.25);
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
}
.savings-big { font-family: 'IBM Plex Mono', monospace; font-size: 2.6rem; font-weight: 600; color: #34d399; }
.savings-label { font-size: 0.8rem; color: #64748b; margin-top: 0.3rem; }

/* Backlog table */
.bl-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.bl-table th {
    background: rgba(56,189,248,0.1);
    color: #38bdf8;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.5rem 0.7rem;
    text-align: left;
}
.bl-table td { padding: 0.45rem 0.7rem; border-bottom: 1px solid rgba(255,255,255,0.04); color: #cbd5e1; }
.bl-table tr:last-child td { border-bottom: none; }
.bl-table tr:hover td { background: rgba(255,255,255,0.03); }

/* Badge */
.badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 999px; font-size: 0.68rem; font-family: 'IBM Plex Mono', monospace; }
.b-crit   { background: rgba(248,113,113,0.15); color: #f87171; }
.b-high   { background: rgba(251,191,36,0.15);  color: #fbbf24; }
.b-med    { background: rgba(56,189,248,0.15);   color: #38bdf8; }
.b-low    { background: rgba(52,211,153,0.15);   color: #34d399; }
.b-vlow   { background: rgba(167,139,250,0.15);  color: #a78bfa; }

.stButton > button {
    background: linear-gradient(135deg, #0284c7, #4f46e5);
    color: white; border: none; border-radius: 10px;
    padding: 0.6rem 1.8rem; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem; font-weight: 600; width: 100%;
}
.stButton > button:hover { opacity: 0.85; }

hr { border-color: rgba(255,255,255,0.06) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATPLOTLIB THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1424", "axes.facecolor": "#0d1424",
    "axes.edgecolor": "#1e293b", "axes.labelcolor": "#64748b",
    "xtick.color": "#475569", "ytick.color": "#475569",
    "text.color": "#c8d4e8", "grid.color": "#1e293b",
    "font.family": "monospace",
})

SEVERITY_COLORS = {
    "Critical":  "#f87171",
    "High":      "#fbbf24",
    "Medium":    "#38bdf8",
    "Low":       "#34d399",
    "Very Low":  "#a78bfa",
}

# ─────────────────────────────────────────────
# TIME MATRIX  (work type i × technician type j)
# ─────────────────────────────────────────────
t_matrix = {
    (1,1):3.0,(1,2):2.6,(1,3):2.5,(1,4):2.2,(1,5):2.0,
    (2,1):4.0,(2,2):3.4,(2,3):3.25,(2,4):2.8,(2,5):2.5,
    (3,1):5.0,(3,2):4.2,(3,3):4.0,(3,4):3.4,(3,5):3.0,
    (4,1):6.0,(4,2):5.0,(4,3):4.75,(4,4):4.0,(4,5):3.5,
    (5,1):7.0,(5,2):5.8,(5,3):5.5,(5,4):4.6,(5,5):4.0,
}

SEVERITY_ORDER = ["Critical", "High", "Medium", "Low", "Very Low"]
COST_PER_JOB   = 10   # $

# ─────────────────────────────────────────────
# HELPER: SINGLE-DAY GUROBI OPTIMISATION
# ─────────────────────────────────────────────
def optimise_day(fibre_demand: dict, copper_demand: dict, workers: dict, H: int):
    """
    fibre_demand  = {1..5: int}  (indexed by severity as work-type)
    copper_demand = {1..5: int}
    workers       = {1..5: int}  (technician type counts)
    Returns (fibre_done, copper_done, status)
    """
    I = range(1, 6)
    J = range(1, 6)

    model = gp.Model("day")
    model.setParam("OutputFlag", 0)

    xF = model.addVars(I, J, vtype=GRB.INTEGER, name="F", lb=0)
    xC = model.addVars(I, J, vtype=GRB.INTEGER, name="C", lb=0)

    # Objective: maximise total jobs (Fibre priority)
    model.setObjective(
        gp.quicksum(xF[i,j] for i in I for j in J)
        + 0.8 * gp.quicksum(xC[i,j] for i in I for j in J),
        GRB.MAXIMIZE
    )

    # Demand upper bounds
    for i in I:
        model.addConstr(gp.quicksum(xF[i,j] for j in J) <= fibre_demand[i])
        model.addConstr(gp.quicksum(xC[i,j] for j in J) <= copper_demand[i])

    # Capacity per technician type
    for j in J:
        model.addConstr(
            gp.quicksum(t_matrix[i,j] * (xF[i,j] + xC[i,j]) for i in I)
            <= workers[j] * H
        )

    model.optimize()

    if model.status == GRB.OPTIMAL:
        fd = {i: int(sum(xF[i,j].X for j in J)) for i in I}
        cd = {i: int(sum(xC[i,j].X for j in J)) for i in I}
        return fd, cd, "optimal"
    else:
        return {i:0 for i in I}, {i:0 for i in I}, "infeasible"


# ─────────────────────────────────────────────
# HELPER: RANDOM BASELINE (non-data-driven)
# ─────────────────────────────────────────────
def random_baseline(fibre_demand: dict, copper_demand: dict, workers: dict, H: int):
    """Naive FIFO proportional allocation without optimisation."""
    I = range(1, 6)
    total_capacity = sum(workers[j] * H for j in range(1, 6))
    # Average time per job ≈ 4 hrs (mid-range)
    avg_time = 4.0
    max_jobs  = int(total_capacity / avg_time)

    fd = {}
    cd = {}
    remaining = max_jobs
    for i in I:
        take = min(fibre_demand[i], remaining // 2)
        fd[i] = take
        remaining -= take
    for i in I:
        take = min(copper_demand[i], remaining // 2)
        cd[i] = take
        remaining -= take

    return fd, cd


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <div style="font-size:2.5rem">📅</div>
  <div>
    <h1>30-Day Technician Allocation Optimizer</h1>
    <p>Upload forecasted demand · Configure technicians · Run Gurobi · See savings vs random allocation</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Upload Forecast")
    uploaded = st.file_uploader(
        "30-day forecast Excel file",
        type=["xlsx", "xls"],
        help="Columns: Forecasted_day_index, severity_Critical, severity_High, "
             "severity_Medium, severity_Low, severity_Very Low, "
             "severity_Critical_new, severity_High_new, severity_Medium_new, "
             "severity_Low_new, severity_Very Low_new"
    )

    st.markdown("### 👷 Technicians Available")
    H = st.number_input("Work hours / day", min_value=1, max_value=24, value=8)
    workers = {
        j: st.number_input(f"Type {j}", min_value=0, value=10, key=f"w{j}")
        for j in range(1, 6)
    }

    st.markdown("### 💰 Cost Settings")
    cost_per_job = st.number_input("Charge per job ($)", min_value=1, value=COST_PER_JOB)

    st.markdown("---")
    run = st.button("🚀 Run 30-Day Optimisation")

# ─────────────────────────────────────────────
# IDLE STATE
# ─────────────────────────────────────────────
if not run or uploaded is None:
    if not run and uploaded is None:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#1e293b;">
          <div style="font-size:3.5rem;margin-bottom:1rem">📊</div>
          <div style="font-family:'IBM Plex Mono',monospace;color:#334155;font-size:0.95rem;">
            Upload your 30-day forecast Excel and click Run
          </div>
          <div style="color:#1e293b;font-size:0.8rem;margin-top:0.5rem;">
            Expected columns: Forecasted_day_index + 10 severity columns
          </div>
        </div>""", unsafe_allow_html=True)
    elif uploaded is None:
        st.warning("⚠️ Please upload the forecast Excel file first.")
    st.stop()

# ─────────────────────────────────────────────
# LOAD & VALIDATE EXCEL
# ─────────────────────────────────────────────
try:
    df_raw = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"❌ Could not read Excel: {e}")
    st.stop()

required_cols = [
    "Forecasted_day_index",
    "severity_Critical","severity_High","severity_Medium","severity_Low","severity_Very Low",
    "severity_Critical_new","severity_High_new","severity_Medium_new",
    "severity_Low_new","severity_Very Low_new",
]
missing = [c for c in required_cols if c not in df_raw.columns]
if missing:
    st.error(f"❌ Missing columns: {missing}")
    st.stop()

df_raw = df_raw[required_cols].fillna(0)
df_raw[required_cols[1:]] = df_raw[required_cols[1:]].astype(int)
n_days = len(df_raw)

# ─────────────────────────────────────────────
# RUN OPTIMISATION DAY BY DAY
# ─────────────────────────────────────────────
I = list(range(1, 6))   # work types == severity levels

progress_bar = st.progress(0, text="Optimising day 1 …")

results = []            # one dict per day
backlog_fibre  = {i: 0 for i in I}   # carry-over from prev days
backlog_copper = {i: 0 for i in I}

for idx, row in df_raw.iterrows():
    day_num = int(row["Forecasted_day_index"])

    # Fibre demand = forecasted + backlog from yesterday
    fibre_demand = {
        i: int(row[f"severity_{sev}"]) + backlog_fibre[i]
        for i, sev in enumerate(SEVERITY_ORDER, 1)
    }
    copper_demand = {
        i: int(row[f"severity_{sev}_new"]) + backlog_copper[i]
        for i, sev in enumerate(SEVERITY_ORDER, 1)
    }

    # Optimised
    fd_opt, cd_opt, status = optimise_day(fibre_demand, copper_demand, workers, H)

    # Baseline (random)
    fd_rnd, cd_rnd = random_baseline(fibre_demand, copper_demand, workers, H)

    total_opt = sum(fd_opt.values()) + sum(cd_opt.values())
    total_rnd = sum(fd_rnd.values()) + sum(cd_rnd.values())

    # Unmet → next day backlog
    new_bl_fibre  = {i: max(0, fibre_demand[i]  - fd_opt[i]) for i in I}
    new_bl_copper = {i: max(0, copper_demand[i] - cd_opt[i]) for i in I}

    results.append({
        "day":           day_num,
        "fibre_demand":  sum(fibre_demand.values()),
        "copper_demand": sum(copper_demand.values()),
        "opt_fibre":     sum(fd_opt.values()),
        "opt_copper":    sum(cd_opt.values()),
        "opt_total":     total_opt,
        "rnd_total":     total_rnd,
        "extra_jobs":    total_opt - total_rnd,
        "backlog_carry": sum(new_bl_fibre.values()) + sum(new_bl_copper.values()),
        "bl_f_detail":   new_bl_fibre,
        "bl_c_detail":   new_bl_copper,
        "status":        status,
        # raw demands (with backlog included)
        "f_demand_detail":  fibre_demand,
        "c_demand_detail":  copper_demand,
    })

    # Update backlogs
    backlog_fibre  = new_bl_fibre
    backlog_copper = new_bl_copper

    progress_bar.progress((idx + 1) / n_days, text=f"Optimising day {day_num} of {n_days} …")

progress_bar.empty()

df_res = pd.DataFrame(results)

# ─────────────────────────────────────────────
# AGGREGATE KPIs
# ─────────────────────────────────────────────
total_opt_jobs  = df_res["opt_total"].sum()
total_rnd_jobs  = df_res["rnd_total"].sum()
extra_jobs      = total_opt_jobs - total_rnd_jobs
monthly_savings = extra_jobs * cost_per_job
annual_savings  = monthly_savings * 12
total_demand    = df_res["fibre_demand"].sum() + df_res["copper_demand"].sum()
total_backlog   = df_res["backlog_carry"].sum()
avg_utilisation = total_opt_jobs / total_demand if total_demand > 0 else 0

# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
st.markdown('<div class="sec-title">30-Day Performance Summary</div>', unsafe_allow_html=True)

def kpi_card(label, value, cls, sub=""):
    return f"""<div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {cls}">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""

st.markdown(f"""
<div class="kpi-grid">
  {kpi_card("Opt. Jobs Done",  f"{total_opt_jobs:,}", "kpi-blue",   f"of {total_demand:,} demand")}
  {kpi_card("Extra vs Random", f"+{extra_jobs:,}",    "kpi-green",  "jobs gained")}
  {kpi_card("Monthly Savings", f"${monthly_savings:,.0f}", "kpi-green", f"@ ${cost_per_job}/job")}
  {kpi_card("Annual Savings",  f"${annual_savings:,.0f}",  "kpi-amber", "× 12 months")}
  {kpi_card("Total Backlog",   f"{total_backlog:,}",  "kpi-red",    "unfulfilled jobs")}
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SAVINGS HIGHLIGHT
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="savings-box">
  <div style="font-size:0.75rem;font-family:'IBM Plex Mono',monospace;color:#64748b;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.5rem;">
    Annual cost savings vs non-data-driven allocation
  </div>
  <div class="savings-big">${annual_savings:,.0f}</div>
  <div class="savings-label">
    {extra_jobs:,} extra jobs/month × ${cost_per_job}/job × 12 months
    &nbsp;·&nbsp; Optimiser completes <b style="color:#38bdf8">{(total_opt_jobs/max(total_rnd_jobs,1)-1)*100:.1f}%</b> more jobs than random allocation
  </div>
</div>
<br>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHARTS  — row 1
# ─────────────────────────────────────────────
st.markdown('<div class="sec-title">Daily Trends</div>', unsafe_allow_html=True)
col1, col2 = st.columns([1.6, 1])

with col1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    days = df_res["day"].tolist()
    ax.fill_between(days, df_res["rnd_total"], alpha=0.25, color="#64748b", label="Random allocation")
    ax.plot(days, df_res["rnd_total"], color="#64748b", linewidth=1, linestyle="--")
    ax.fill_between(days, df_res["opt_total"], alpha=0.3, color="#38bdf8", label="Optimised")
    ax.plot(days, df_res["opt_total"], color="#38bdf8", linewidth=1.8)
    ax.fill_between(days, df_res["rnd_total"], df_res["opt_total"],
                    alpha=0.15, color="#34d399", label="Extra jobs (gain)")
    ax.set_xlabel("Day", fontsize=9)
    ax.set_ylabel("Jobs completed", fontsize=9)
    ax.set_title("Optimised vs Random — Daily jobs", fontsize=10, color="#94a3b8", pad=8)
    ax.legend(fontsize=8, framealpha=0.15)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(4.5, 3.4))
    ax2.bar(days, df_res["backlog_carry"], color="#f87171", alpha=0.8, width=0.7)
    ax2.set_xlabel("Day", fontsize=9)
    ax2.set_ylabel("Backlog jobs", fontsize=9)
    ax2.set_title("Daily carry-over backlog", fontsize=10, color="#94a3b8", pad=8)
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines[["top","right"]].set_visible(False)
    fig2.tight_layout()
    st.pyplot(fig2)
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHARTS  — row 2
# ─────────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(5.5, 3.2))
    ax3.bar(days, df_res["opt_fibre"],  color="#38bdf8", alpha=0.85, label="Fibre")
    ax3.bar(days, df_res["opt_copper"], color="#34d399", alpha=0.85,
            bottom=df_res["opt_fibre"], label="Copper")
    ax3.set_xlabel("Day", fontsize=9)
    ax3.set_ylabel("Jobs", fontsize=9)
    ax3.set_title("Fibre vs Copper jobs completed per day", fontsize=10, color="#94a3b8", pad=8)
    ax3.legend(fontsize=8, framealpha=0.15)
    ax3.grid(axis="y", alpha=0.3)
    ax3.spines[["top","right"]].set_visible(False)
    fig3.tight_layout()
    st.pyplot(fig3)
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    fig4, ax4 = plt.subplots(figsize=(5.5, 3.2))
    extra = df_res["extra_jobs"].tolist()
    colors4 = ["#34d399" if v >= 0 else "#f87171" for v in extra]
    ax4.bar(days, extra, color=colors4, alpha=0.85, width=0.7)
    ax4.axhline(0, color="#475569", linewidth=0.8)
    ax4.set_xlabel("Day", fontsize=9)
    ax4.set_ylabel("Extra jobs", fontsize=9)
    ax4.set_title("Daily gain: Optimised − Random", fontsize=10, color="#94a3b8", pad=8)
    ax4.grid(axis="y", alpha=0.3)
    ax4.spines[["top","right"]].set_visible(False)
    fig4.tight_layout()
    st.pyplot(fig4)
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# BACKLOG DETAIL TABLE
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="sec-title">Backlog Detail — Day-by-Day Carry-Over</div>', unsafe_allow_html=True)

st.info(
    "🔁 **How backlog works:** Jobs not completed on Day N are carried forward as backlog "
    "and added to Day N+1 demand. The table below shows how many backlog jobs (by severity) "
    "were rolled into each day's queue. A day with 0 backlog means the previous day was fully cleared.",
    icon=None
)

# Build backlog display: for each day, show how much backlog came *from* the prior days
bl_rows = []
running_bl = {sev: 0 for sev in SEVERITY_ORDER}

for r in results:
    day = r["day"]
    # Backlog that was *added* to today = carried from yesterday
    # f_demand_detail includes yesterday's backlog already
    f_today   = r["f_demand_detail"]
    c_today   = r["c_demand_detail"]

    # Compute fresh demand (no backlog) from raw df
    raw_row = df_raw[df_raw["Forecasted_day_index"] == day].iloc[0]
    fresh_f = {i: int(raw_row[f"severity_{sev}"]) for i, sev in enumerate(SEVERITY_ORDER, 1)}
    fresh_c = {i: int(raw_row[f"severity_{sev}_new"]) for i, sev in enumerate(SEVERITY_ORDER, 1)}

    backlog_f = {i: f_today[i] - fresh_f[i] for i in I}
    backlog_c = {i: c_today[i] - fresh_c[i] for i in I}

    total_bl_in = sum(backlog_f.values()) + sum(backlog_c.values())

    bl_rows.append({
        "Day": day,
        "Backlog in (total)": total_bl_in,
        "BL Critical": backlog_f[1] + backlog_c[1],
        "BL High":     backlog_f[2] + backlog_c[2],
        "BL Medium":   backlog_f[3] + backlog_c[3],
        "BL Low":      backlog_f[4] + backlog_c[4],
        "BL Very Low": backlog_f[5] + backlog_c[5],
        "Backlog out (unfulfilled)": r["backlog_carry"],
        "Opt Jobs": r["opt_total"],
        "Rnd Jobs": r["rnd_total"],
        "Gain": r["extra_jobs"],
    })

df_bl = pd.DataFrame(bl_rows)

# Render scrollable table
rows_html = ""
for _, r in df_bl.iterrows():
    gain_color = "#34d399" if r["Gain"] >= 0 else "#f87171"
    bl_in_color = "#fbbf24" if r["Backlog in (total)"] > 0 else "#34d399"
    bl_out_color = "#f87171" if r["Backlog out (unfulfilled)"] > 0 else "#34d399"
    rows_html += f"""<tr>
        <td><b>Day {int(r['Day'])}</b></td>
        <td style="color:{bl_in_color}">{int(r['Backlog in (total)'])}</td>
        <td><span class="badge b-crit">{int(r['BL Critical'])}</span></td>
        <td><span class="badge b-high">{int(r['BL High'])}</span></td>
        <td><span class="badge b-med">{int(r['BL Medium'])}</span></td>
        <td><span class="badge b-low">{int(r['BL Low'])}</span></td>
        <td><span class="badge b-vlow">{int(r['BL Very Low'])}</span></td>
        <td style="color:{bl_out_color}">{int(r['Backlog out (unfulfilled)'])}</td>
        <td style="color:#38bdf8">{int(r['Opt Jobs'])}</td>
        <td style="color:#64748b">{int(r['Rnd Jobs'])}</td>
        <td style="color:{gain_color};font-weight:600">+{int(r['Gain'])}</td>
    </tr>"""

st.markdown(f"""
<div class="panel" style="overflow-x:auto;max-height:420px;overflow-y:auto;">
<table class="bl-table">
  <thead><tr>
    <th>Day</th>
    <th>Backlog In</th>
    <th>BL Critical</th><th>BL High</th><th>BL Medium</th><th>BL Low</th><th>BL Very Low</th>
    <th>Backlog Out</th>
    <th>Opt Jobs</th><th>Rnd Jobs</th><th>Gain</th>
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CUMULATIVE SAVINGS CHART
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="sec-title">Cumulative Savings Trajectory</div>', unsafe_allow_html=True)
st.markdown('<div class="panel">', unsafe_allow_html=True)

cumul_savings = (df_res["extra_jobs"] * cost_per_job).cumsum()
fig5, ax5 = plt.subplots(figsize=(12, 3))
ax5.fill_between(days, cumul_savings, alpha=0.25, color="#34d399")
ax5.plot(days, cumul_savings, color="#34d399", linewidth=2)
ax5.axhline(monthly_savings, color="#fbbf24", linewidth=1, linestyle="--",
            label=f"Monthly total: ${monthly_savings:,.0f}")
for spine in ["top","right"]: ax5.spines[spine].set_visible(False)
ax5.set_xlabel("Day", fontsize=9)
ax5.set_ylabel("Cumulative savings ($)", fontsize=9)
ax5.set_title("Running cost savings (Optimised vs Random)", fontsize=10, color="#94a3b8", pad=8)
ax5.legend(fontsize=9, framealpha=0.15)
ax5.grid(axis="y", alpha=0.3)
fig5.tight_layout()
st.pyplot(fig5)
st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DOWNLOAD RESULTS
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
export_df = df_bl.copy()
export_df["Monthly Savings ($)"] = ""
export_df.at[0, "Monthly Savings ($)"] = monthly_savings
export_df["Annual Savings ($)"] = ""
export_df.at[0, "Annual Savings ($)"] = annual_savings

buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as writer:
    export_df.to_excel(writer, index=False, sheet_name="Results")
    df_raw.to_excel(writer, index=False, sheet_name="Input Forecast")
buf.seek(0)

st.download_button(
    "⬇️  Download Full Results (Excel)",
    data=buf,
    file_name="optimisation_results_30day.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#1e293b;font-family:'IBM Plex Mono',monospace;
            font-size:0.65rem;padding:1.5rem;margin-top:1rem;">
  GUROBI ILP · 30-DAY ROLLING BACKLOG · SEVERITY-WEIGHTED DEMAND
</div>
""", unsafe_allow_html=True)
