import matplotlib.pyplot as plt
from run_experiment import run_experiment

# =========================
# CONFIG (REALISTIC)
# =========================
CONFIG = {
    "window": 5,
    "capacity_per_instance": 5,
    "max_instances": 2000,
    "max_delta": 25,
    "instance_cost": 0.02,
    "under_penalty": 6
}

STATIC_INSTANCES = 1200

# =========================
# RUN EXPERIMENT
# =========================
results = run_experiment(CONFIG)

df = results["df"]
ort_allocation = results["ort_allocation"]

# =========================
# BUILD CAPACITY COLUMNS
# =========================

# OR-Tools
df["ort_capacity"] = [
    x * CONFIG["capacity_per_instance"]
    for x in ort_allocation
]

# Static
df["static_capacity"] = STATIC_INSTANCES * CONFIG["capacity_per_instance"]

# =========================
# GRAPH FUNCTION
# =========================
def plot_policy(name, capacity_col):
    plt.figure(figsize=(12, 6))

    demand = df["cpu_demand"]
    capacity = df[capacity_col]

    plt.plot(demand, label="Demand", linewidth=2.5)
    plt.plot(capacity, label=f"{name} Capacity")

    # UNDER
    plt.fill_between(
        range(len(df)),
        demand,
        capacity,
        where=(demand > capacity),
        alpha=0.3,
        label="Under"
    )

    # OVER
    plt.fill_between(
        range(len(df)),
        demand,
        capacity,
        where=(capacity > demand),
        alpha=0.3,
        label="Over"
    )

    plt.title(f"{name} Policy: Demand vs Capacity")
    plt.xlabel("Time Step")
    plt.ylabel("CPU Demand")
    plt.legend(loc="upper left")
    plt.grid()

    plt.show()


# =========================
# INDIVIDUAL POLICY PLOTS
# =========================

plot_policy("Rolling", "rolling_capacity")
plot_policy("Target Tracking", "tt_capacity")
plot_policy("ML", "ml_capacity")
plot_policy("Hybrid", "hybrid_capacity")
plot_policy("OR-Tools", "ort_capacity")
plot_policy("Static", "static_capacity")

# =========================
# COMBINED GRAPH
# =========================
plt.figure(figsize=(12, 6))

plt.plot(df["cpu_demand"], label="Demand", linewidth=2.5)

plt.plot(df["rolling_capacity"], label="Rolling")
plt.plot(df["tt_capacity"], label="Target Tracking")
plt.plot(df["ml_capacity"], label="ML")
plt.plot(df["hybrid_capacity"], label="Hybrid")
plt.plot(df["ort_capacity"], label="OR-Tools")
plt.plot(df["static_capacity"], label="Static", linestyle="--", alpha=0.7)

plt.title("All Policies Comparison")
plt.xlabel("Time Step")
plt.ylabel("CPU Demand")
plt.legend(loc="upper left")
plt.grid()

plt.show()