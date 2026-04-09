import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Technician Allocation Optimizer",
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
    }
    .kpi-card:hover { border-color: rgba(167,139,250,0.4); }
    .kpi-label {
        font-size: 0.72rem;
        font-family: 'Space Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748b;
        margin-bottom: 0.4rem;
    }
    .kpi-value { font-size: 2.2rem; font-weight: 700; color: #a78bfa; line-height: 1; }
    .kpi-sub { font-size: 0.78rem; color: #475569; margin-top: 0.3rem; }

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

    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 2rem;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    .styled-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    .styled-table th {
        background: rgba(167,139,250,0.15);
        color: #a78bfa;
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 0.6rem 0.8rem;
        text-align: left;
    }
    .styled-table td {
        padding: 0.55rem 0.8rem;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        color: #e2e8f0;
    }
    .styled-table tr:last-child td { border-bottom: none; }

    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.7rem;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
    }
    .badge-ok    { background: rgba(16,185,129,0.15);  color: #10b981; }
    .badge-warn  { background: rgba(245,158,11,0.15);  color: #f59e0b; }
    .badge-error { background: rgba(239,68,68,0.15);   color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor":   "#1a1a2e",
    "axes.edgecolor":   "#2d2d4e",
    "axes.labelcolor":  "#94a3b8",
    "xtick.color":      "#64748b",
    "ytick.color":      "#64748b",
    "text.color":       "#e0e0f0",
    "grid.color":       "#2d2d4e",
    "font.family":      "monospace",
})

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <h1>🔧 Technician Allocation Optimizer</h1>
  <p>SciPy MILP-powered workforce scheduling · Maximize job throughput across Fibre, Copper & Backlog</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    H = st.number_input("Work hours per day", min_value=1, max_value=24, value=8)

    st.markdown("### 👷 Technicians Available")
    workers = {
        j: st.number_input(f"Type {j}", min_value=0, value=5, key=f"w{j}")
        for j in range(1, 6)
    }

    st.markdown("### 📋 Demand by Work Type")

    def demand_block(title, prefix, defaults):
        st.markdown(f"**{title}**")
        return {
            i: st.number_input(f"Work {i}", min_value=0, value=defaults[i-1], key=f"{prefix}{i}")
            for i in range(1, 6)
        }

    F = demand_block("🔵 Fibre",         "F", [10,10,10,10,10])
    B = demand_block("🟠 Backlog Copper", "B", [10,10,10,10,10])
    C = demand_block("🟢 New Copper",     "C", [10,10,10,10,10])

    st.markdown("### 🎛️ Priority Weight")
    beta = st.slider("New Copper Weight (β)", 0.0, 1.0, 0.5, 0.05)
    st.caption("Higher β = more priority to New Copper jobs")

    st.markdown("---")
    run = st.button("🚀 Run Optimization")

# ─────────────────────────────────────────────
# TIME MATRIX & INDEX HELPERS
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
N = 75  # 3 job types × 5 work types × 5 tech types

# Variable layout: [xF(0-24), xB(25-49), xC(50-74)]
# Index within each block: (i-1)*5 + (j-1)
def idxF(i, j): return (i-1)*5 + (j-1)
def idxB(i, j): return 25 + (i-1)*5 + (j-1)
def idxC(i, j): return 50 + (i-1)*5 + (j-1)

# ─────────────────────────────────────────────
# IDLE STATE
# ─────────────────────────────────────────────
if not run:
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; color:#334155;">
        <div style="font-size:4rem; margin-bottom:1rem;">⚡</div>
        <div style="font-family:'Space Mono',monospace; font-size:1rem; color:#475569;">
            Configure inputs in the sidebar and click Run Optimization
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# BUILD & SOLVE WITH SCIPY MILP
# ─────────────────────────────────────────────
with st.spinner("Running SciPy MILP optimizer..."):

    # ── Objective (scipy minimizes, so negate) ──
    c = np.zeros(N)
    for i in I:
        for j in J:
            c[idxF(i,j)] = -1.0        # Fibre
            c[idxB(i,j)] = -1.0        # Backlog
            c[idxC(i,j)] = -beta       # Copper (weighted)

    # ── Constraint matrix ────────────────────────
    # Total constraints: 5(F demand)+5(B demand)+5(C demand)+5(capacity)+5(copper window) = 25
    A = np.zeros((25, N))
    b_upper = np.zeros(25)
    b_lower = np.full(25, -np.inf)   # all are <= constraints

    row = 0

    # 1. Fibre demand: sum_j xF[i,j] <= F[i]
    for i in I:
        for j in J:
            A[row, idxF(i,j)] = 1.0
        b_upper[row] = F[i]
        row += 1

    # 2. Backlog demand: sum_j xB[i,j] <= B[i]
    for i in I:
        for j in J:
            A[row, idxB(i,j)] = 1.0
        b_upper[row] = B[i]
        row += 1

    # 3. Copper demand: sum_j xC[i,j] <= C[i]
    for i in I:
        for j in J:
            A[row, idxC(i,j)] = 1.0
        b_upper[row] = C[i]
        row += 1

    # 4. Capacity: sum_i t[i,j]*(xF+xB+xC)[i,j] <= workers[j]*H
    for j in J:
        for i in I:
            A[row, idxF(i,j)] = t[i,j]
            A[row, idxB(i,j)] = t[i,j]
            A[row, idxC(i,j)] = t[i,j]
        b_upper[row] = workers[j] * H
        row += 1

    # 5. Copper 2-day window: sum_i t[i,j]*(xB+xC)[i,j] <= 2*workers[j]*H
    for j in J:
        for i in I:
            A[row, idxB(i,j)] = t[i,j]
            A[row, idxC(i,j)] = t[i,j]
        b_upper[row] = 2 * workers[j] * H
        row += 1

    constraints = LinearConstraint(A, b_lower, b_upper)

    # ── Bounds: all variables >= 0 ───────────────
    bounds = Bounds(lb=np.zeros(N), ub=np.full(N, np.inf))

    # ── Integrality: 1 = integer for all variables ─
    integrality = np.ones(N)

    result = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)

# ─────────────────────────────────────────────
# CHECK RESULT
# ─────────────────────────────────────────────
if not result.success:
    st.error(f"❌ Optimization failed: {result.message}. Try relaxing constraints or increasing technician counts.")
    st.stop()

x = result.x  # solution vector

# ─────────────────────────────────────────────
# EXTRACT VALUES
# ─────────────────────────────────────────────
def get(i, j, offset_fn):
    return max(0, int(round(x[offset_fn(i,j)])))

total_fibre   = sum(get(i,j,idxF) for i in I for j in J)
total_backlog = sum(get(i,j,idxB) for i in I for j in J)
total_copper  = sum(get(i,j,idxC) for i in I for j in J)
total_jobs    = total_fibre + total_backlog + total_copper

utilization = []
for j in J:
    used = sum(t[i,j] * (get(i,j,idxF) + get(i,j,idxB) + get(i,j,idxC)) for i in I)
    cap  = workers[j] * H
    utilization.append(used / cap if cap > 0 else 0)

tech_labels = [f"T{j}" for j in J]

# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Key Performance Indicators</div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)

def kpi(col, label, value, sub=""):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

kpi(k1, "Total Jobs",      total_jobs,    "completed")
kpi(k2, "Fibre Jobs",      total_fibre,   f"of {sum(F.values())} demand")
kpi(k3, "Backlog Jobs",    total_backlog, f"of {sum(B.values())} demand")
kpi(k4, "New Copper Jobs", total_copper,  f"of {sum(C.values())} demand")
kpi(k5, "Avg Utilization", f"{np.mean(utilization):.0%}", "across all types")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHARTS ROW
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Allocation Analysis</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1.6, 1.4])

# ── Donut pie ────────────────────────────────
with col1:
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.markdown("**Job Mix Distribution**")
    fig1, ax1 = plt.subplots(figsize=(4, 3.5))
    wedge_colors = ["#60a5fa", "#f59e0b", "#34d399"]
    wedges, texts, autotexts = ax1.pie(
        [total_fibre, total_backlog, total_copper],
        labels=["Fibre", "Backlog", "Copper"],
        colors=wedge_colors,
        autopct="%1.0f%%",
        startangle=140,
        wedgeprops=dict(width=0.65, edgecolor="#1a1a2e", linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(9); at.set_color("white")
    ax1.set_title("Job Categories", fontsize=10, color="#94a3b8", pad=12)
    fig1.tight_layout()
    st.pyplot(fig1)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Utilization bar chart ─────────────────────
with col2:
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.markdown("**Technician Utilization**")
    fig2, ax2 = plt.subplots(figsize=(5.5, 3.5))
    bar_colors = [
        "#ef4444" if u > 0.9 else "#f59e0b" if u > 0.7 else "#34d399"
        for u in utilization
    ]
    bars = ax2.bar(tech_labels, utilization, color=bar_colors, width=0.55, zorder=3)
    ax2.axhline(1.0, color="#ef4444", linestyle="--", linewidth=1, alpha=0.5, label="Capacity")
    ax2.axhline(0.7, color="#f59e0b", linestyle=":",  linewidth=1, alpha=0.4, label="70% threshold")
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("Utilization", fontsize=9)
    ax2.set_title("Load per Technician Type", fontsize=10, color="#94a3b8", pad=10)
    ax2.legend(fontsize=8, framealpha=0.2)
    ax2.grid(axis="y", zorder=0)
    for bar, u in zip(bars, utilization):
        ax2.text(bar.get_x() + bar.get_width()/2, u + 0.02, f"{u:.0%}",
                 ha="center", va="bottom", fontsize=8.5, color="white")
    ax2.spines[["top","right"]].set_visible(False)
    fig2.tight_layout()
    st.pyplot(fig2)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Demand vs Completed bar chart ─────────────
with col3:
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.markdown("**Demand vs Completed**")
    fig3, ax3 = plt.subplots(figsize=(4.5, 3.5))
    categories = ["Fibre", "Backlog", "Copper"]
    demand     = [sum(F.values()), sum(B.values()), sum(C.values())]
    completed  = [total_fibre, total_backlog, total_copper]
    x_pos = np.arange(len(categories))
    ax3.bar(x_pos - 0.2, demand,    0.35, label="Demand",    color="#334155", zorder=3)
    ax3.bar(x_pos + 0.2, completed, 0.35, label="Completed",
            color=["#60a5fa","#f59e0b","#34d399"], zorder=3)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(categories, fontsize=9)
    ax3.set_title("Fulfillment vs Demand", fontsize=10, color="#94a3b8", pad=10)
    ax3.legend(fontsize=8, framealpha=0.2)
    ax3.grid(axis="y", zorder=0)
    ax3.spines[["top","right"]].set_visible(False)
    fig3.tight_layout()
    st.pyplot(fig3)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DETAIL TABLE + INSIGHTS
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Detailed Breakdown & Insights</div>', unsafe_allow_html=True)

tab1, tab2 = st.columns([1.8, 1])

# ── Per-technician table ──────────────────────
with tab1:
    rows = []
    for j in J:
        f_jobs = sum(get(i,j,idxF) for i in I)
        b_jobs = sum(get(i,j,idxB) for i in I)
        c_jobs = sum(get(i,j,idxC) for i in I)
        total  = f_jobs + b_jobs + c_jobs
        util   = utilization[j-1]
        badge  = (
            '<span class="badge badge-error">Overloaded</span>' if util > 0.9 else
            '<span class="badge badge-warn">Moderate</span>'    if util > 0.7 else
            '<span class="badge badge-ok">Efficient</span>'
        )
        rows.append(f"""
        <tr>
            <td><b>Type {j}</b></td>
            <td>{workers[j]}</td>
            <td>{f_jobs}</td><td>{b_jobs}</td><td>{c_jobs}</td>
            <td>{total}</td><td>{util:.0%}</td><td>{badge}</td>
        </tr>""")

    st.markdown(f"""
    <div class="chart-panel">
    <table class="styled-table">
        <thead><tr>
            <th>Technician</th><th>Workers</th>
            <th>Fibre</th><th>Backlog</th><th>Copper</th>
            <th>Total Jobs</th><th>Utilization</th><th>Status</th>
        </tr></thead>
        <tbody>{''.join(rows)}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

# ── Insights ──────────────────────────────────
with tab2:
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.markdown("**🧠 Auto Insights**")

    insights = []
    max_util     = max(utilization)
    min_util     = min(utilization)
    bottleneck_j = tech_labels[utilization.index(max_util)]
    underload_j  = tech_labels[utilization.index(min_util)]

    if max_util > 0.9:
        insights.append(("warn",  f"⚠️ **{bottleneck_j}** is near capacity at {max_util:.0%}"))
    if max_util > 0.95:
        insights.append(("error", f"🚀 Add more **{bottleneck_j}** technicians to avoid missed jobs"))
    if min_util < 0.5:
        insights.append(("info",  f"💡 **{underload_j}** is underutilized ({min_util:.0%}) — reassign or reduce"))

    for i in I:
        done   = sum((get(i,j,idxF) + get(i,j,idxB) + get(i,j,idxC)) for j in J)
        demand = F[i] + B[i] + C[i]
        if done < demand:
            insights.append(("error", f"❌ Work type **{i}**: {int(demand - done)} jobs unmet"))

    if total_fibre + total_backlog < sum(F.values()) + sum(B.values()):
        insights.append(("warn", "⚠️ High-priority Fibre/Backlog jobs not fully completed"))

    if not insights:
        insights.append(("ok", "✅ All operations running optimally! No issues detected."))

    for kind, msg in insights:
        st.markdown(f'<div class="insight-card {kind}">{msg}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#334155; font-family:'Space Mono',monospace; font-size:0.7rem; padding:1rem;">
    SCIPY MILP OPTIMIZER · INTEGER PROGRAMMING · TECHNICIAN SCHEDULING
</div>
""", unsafe_allow_html=True)
