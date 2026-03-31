import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from run_experiment import run_experiment



POLICY_COLORS = {
    "Rolling": "blue",
    "Target Tracking": "green",
    "ML": "purple",
    "Hybrid": "orange",
    "OR-Tools": "black",
}

# ==============================
# FIX NUMPY TYPES
# ==============================
def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj


# ==============================
# UI CONFIG
# ==============================
st.sidebar.header("Config")

# ==============================
# WINDOW
# ==============================
window = st.sidebar.slider("Window", 1, 20, 5)
st.sidebar.caption(
    "Used by: ML, Hybrid, Rolling\n"
    "Controls how many past steps are used to predict demand."
)

# ==============================
# CAPACITY
# ==============================
capacity = st.sidebar.slider("Capacity per Instance", 1, 20, 5)
st.sidebar.caption(
    "Used by: ALL policies\n"
    "Defines how much workload each instance can handle."
)

# ==============================
# MAX INSTANCES
# ==============================
max_instances = st.sidebar.slider("Max Instances", 100, 5000, 2000)
st.sidebar.caption(
    "Used by: OR-Tools, ML, Hybrid\n"
    "Upper limit on how much the system can scale."
)

# ==============================
# MAX DELTA
# ==============================
max_delta = st.sidebar.slider("Max Delta", 1, 100, 25)
st.sidebar.caption(
    "Used by: OR-Tools\n"
    "Limits how quickly scaling can change between time steps."
)

# ==============================
# INSTANCE COST
# ==============================
instance_cost = st.sidebar.slider("Instance Cost", 0.001, 0.1, 0.02)
st.sidebar.caption(
    "Used by: ALL policies\n"
    "Cost of running instances. Affects total cost optimization."
)

# ==============================
# UNDER PENALTY
# ==============================
under_penalty = st.sidebar.slider("Under Penalty", 1, 20, 6)
st.sidebar.caption(
    "Used by: ALL policies (especially OR-Tools)\n"
    "Penalty for not meeting demand (SLA violations)."
)
config = {
    "window": window,
    "capacity_per_instance": capacity,
    "max_instances": max_instances,
    "max_delta": max_delta,
    "instance_cost": instance_cost,
    "under_penalty": under_penalty
}


# ==============================
# RUN SIMULATION
# ==============================
if st.button("Run Simulation"):
    results = run_experiment(config)
    results = convert_numpy(results)
    st.session_state.results = results


# ==============================
# DISPLAY
# ==============================
if "results" in st.session_state:

    results = st.session_state.results
    df = pd.DataFrame(results["df"])

    # ==============================
    # POLICY SELECTOR
    # ==============================
    policies = {
        "Rolling": "rolling_capacity",
        "Target Tracking": "tt_capacity",
        "ML": "ml_capacity",
        "Hybrid": "hybrid_capacity",
        "OR-Tools": "ort_capacity"
    }

    selected_policy = st.selectbox("Select Policy", list(policies.keys()))
    col = policies[selected_policy]

    demand = df["cpu_demand"]

    # ==============================
    # TIME SLIDER (KEY PART)
    # ==============================
    default_t = len(df) // 2  # midpoint
    t = st.slider("Time Step", 10, len(df) - 1, default_t)

    # ==============================
    # PLOT
    # ==============================
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(demand[:t], label="Demand", linewidth=2)
    ax.plot(df[col][:t], label=selected_policy)

    # highlight SLA violations
    ax.fill_between(
        range(t),
        demand[:t],
        df[col][:t],
        where=(demand[:t] > df[col][:t]),
        color="red",
        alpha=0.3,
        label="Under-provision"
    )
    # OVER (light orange) 🔥
    ax.fill_between(
        range(t),
        demand[:t],
        df[col][:t],
        where=(df[col][:t] > demand[:t]),
        color="orange",
        alpha=0.2,
        label="Over-provision"
    )
    ax.axvspan(0, window, color="gray", alpha=0.2, label="Warm-up period")
    ax.set_title(f"{selected_policy} vs Demand")
    ax.legend()
    ax.grid()

    st.pyplot(fig)

    # ==============================
    # LIVE METRICS (VERY COOL)
    # ==============================

    st.markdown("## Policy Metrics Summary")

    policies = ["static", "rolling", "target_tracking", "ml", "hybrid", "ortools"]

    table = []
    for p in policies:
        row = {"policy": p}
        row.update(results[p])
        table.append(row)

    df_metrics = pd.DataFrame(table)

    # ==============================
    # DROP UNUSED COLUMNS
    # ==============================
    df_metrics = df_metrics.drop(columns=["avg_utilization", "peak_utilization"], errors="ignore")

    # ==============================
    # FORMAT VALUES
    # ==============================



    def safe_int(x):
        return f"{int(x):,}" if pd.notna(x) else "-"

    def safe_float(x):
        return f"{x:,.2f}" if pd.notna(x) else "-"

    def safe_percent(x):
        return f"{x:.2f}%" if pd.notna(x) else "-"


    df_metrics["total_cost"] = df_metrics["total_cost"].apply(
        lambda x: f"${safe_float(x)}" if pd.notna(x) else "-"
    )

    df_metrics["total_under"] = df_metrics["total_under"].apply(safe_int)
    df_metrics["total_over"] = df_metrics["total_over"].apply(safe_int)
    df_metrics["stability"] = df_metrics["stability"].apply(safe_int)
    df_metrics["sla_violation_rate"] = df_metrics["sla_violation_rate"].apply(safe_percent)
    # ==============================
    # DISPLAY
    # ==============================
    st.dataframe(df_metrics, use_container_width=True)
