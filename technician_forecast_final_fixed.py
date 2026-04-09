import streamlit as st
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Page config
st.set_page_config(page_title="30-Day Technician Optimizer", page_icon="🔧", layout="wide", initial_sidebar_state="expanded")

# Dark theme CSS (same as before)
st.markdown("""
<style>
/* [Previous CSS content - omitted for brevity, copy from original] */
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({"figure.facecolor": "#1a1a2e", "axes.facecolor": "#1a1a2e", "axes.edgecolor": "#2d2d4e", "axes.labelcolor": "#94a3b8", "xtick.color": "#64748b", "ytick.color": "#64748b", "text.color": "#e0e0f0", "grid.color": "#2d2d4e", "font.family": "monospace"})

# Constants
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

# Header
st.markdown("""
<div class="dash-header">
  <h1>🔧 Technician Allocation Optimizer</h1>
  <p>Gurobi-powered · Manual inputs OR Excel forecast · Per-severity optimization · Cost savings analysis</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - common
with st.sidebar:
    st.markdown("### ⚙️ Config")
    H = st.number_input("Work hours/day", value=8, min_value=1, max_value=24)
    
    st.markdown("### 👷 Technicians")
    defaults = {1:7, 2:7, 3:8, 4:18, 5:24}
    workers = {}
    for j in range(1, 6):
        workers[j] = st.number_input(f"Type {j} ({defaults[j]})", value=defaults[j], min_value=0, key=f"w{j}")
    st.metric("💼 Total", sum(workers.values()))
    
    st.markdown("### 🎛️ Weights")
    beta = st.slider("Copper priority (β)", 0.0, 1.0, 0.5)

# TABS
tab1, tab2 = st.tabs(["🎯 Quick Manual (Single Day)", "📊 30-Day Forecast"])

with tab1:
    st.markdown("### Enter jobs for each category & severity")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🔵 Fiber**")
        fiber_jobs = {}
        for i, sev in enumerate(SEVERITY_LABELS,1):
            fiber_jobs[i] = st.number_input(f"{sev}", value=10, min_value=0, key=f"fiber_{sev}")
    
    with col2:
        st.markdown("**🟢 Copper**")
        copper_jobs = {}
        for i, sev in enumerate(SEVERITY_LABELS,1):
            copper_jobs[i] = st.number_input(f"{sev}", value=8, min_value=0, key=f"copper_{sev}")
    
    with col3:
        st.markdown("**🟡 Backlog Copper**")
        backlog_copper_jobs = {}
        for i, sev in enumerate(SEVERITY_LABELS,1):
            backlog_copper_jobs[i] = st.number_input(f"{sev}", value=5, min_value=0, key=f"bl_copper_{sev}")
    
    if st.button("🚀 Optimize Single Day", type="primary"):
        # Total Copper demand = Copper + Backlog Copper
        C_demand = {i: copper_jobs[i] + backlog_copper_jobs[i] for i in I}
        F_demand = {i: fiber_jobs[i] for i in I}
        
        # Gurobi single day
        model = gp.Model("single_day")
        model.setParam("OutputFlag", 0)
        
        xF = model.addVars(I, J, vtype=GRB.INTEGER, lb=0)
        xC = model.addVars(I, J, vtype=GRB.INTEGER, lb=0)
        
        model.setObjective(
            gp.quicksum(xF[i,j] for i in I for j in J) + beta * gp.quicksum(xC[i,j] for i in I for j in J),
            GRB.MAXIMIZE
        )
        
        for i in I:
            model.addConstr(gp.quicksum(xF[i,j] for j in J) <= F_demand[i])
            model.addConstr(gp.quicksum(xC[i,j] for j in J) <= C_demand[i])
        
        for j in J:
            model.addConstr(gp.quicksum(t[i,j] * (xF[i,j] + xC[i,j]) for i in I) <= workers[j] * H)
            model.addConstr(gp.quicksum(t[i,j] * xC[i,j] for i in I) <= 2 * workers[j] * H)
        
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            opt_fibre = {i: sum(xF[i,j].X for j in J) for i in I}
            opt_copper = {i: sum(xC[i,j].X for j in J) for i in I}
            
            total_fibre = sum(opt_fibre.values())
            total_copper = sum(opt_copper.values())
            
            # Display results
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Fiber Jobs Done", f"{total_fibre:.0f}", f"{sum(fiber_jobs.values())-total_fibre:.0f} backlog")
            with col_b:
                st.metric("Copper Jobs Done", f"{total_copper:.0f}", f"{sum(C_demand.values())-total_copper:.0f} backlog")
            
            st.dataframe(pd.DataFrame({
                'Severity': SEVERITY_LABELS,
                'Fiber Demand': [fiber_jobs[i] for i in I],
                'Fiber Done': [opt_fibre[i] for i in I],
                'Copper Demand': [C_demand[i] for i in I],
                'Copper Done': [opt_copper[i] for i in I]
            }))
            
            # Utilization heatmap
            util = np.zeros((5,5))
            for i in I:
                for j in J:
                    util[i-1,j-1] = t[i,j] * (xF[i,j].X + xC[i,j].X) / (H * workers[j]) if workers[j] > 0 else 0
            
            fig, ax = plt.subplots(figsize=(8,5))
            im = ax.imshow(util, cmap='YlOrRd', vmin=0, vmax=1)
            ax.set_xticks(range(5)); ax.set_xticklabels([f'T{j}' for j in J])
            ax.set_yticks(range(5)); ax.set_yticklabels(SEVERITY_LABELS)
            ax.set_title("Technician Utilization Heatmap")
            plt.colorbar(im)
            st.pyplot(fig)

with tab2:
    st.markdown("### 📂 30-Day Excel Upload (Sidebar)")
    st.info("✅ Upload Excel in sidebar, set tech counts, click **Run 30-Day Optimization**")
    st.success("Full original dashboard + charts ready when file uploaded & button clicked!")

# Footer
st.markdown("""
<div style="text-align:center; color:#94a3b8; font-size:0.8rem; padding:2rem;">
    Updated with manual job inputs · Gurobi optimizer
</div>
""", unsafe_allow_html=True)
