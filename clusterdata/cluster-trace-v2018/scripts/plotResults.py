import matplotlib.pyplot as plt
from baseline_rolling_average import run_experiment

# =========================
# CONFIG
# =========================
CONFIG = {
    "window": 3,
    "capacity_per_instance": 40,
    "max_instances": 250,
    "max_delta": 10,
    "instance_cost": 10,
    "under_penalty": 50
}

STATIC_INSTANCES = 100

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

    # Demand line
    plt.plot(demand, label="Demand", linewidth=2)

    # Capacity line
    plt.plot(capacity, label=f"{name} Capacity")

    # UNDER-PROVISION (red)
    plt.fill_between(
        range(len(df)),
        demand,
        capacity,
        where=(demand > capacity),
        alpha=0.3,
        label="Under-provision",
    )

    # OVER-PROVISION (green)
    plt.fill_between(
        range(len(df)),
        demand,
        capacity,
        where=(capacity > demand),
        alpha=0.3,
        label="Over-provision",
    )

    plt.title(f"{name} Policy: Demand vs Capacity")
    plt.xlabel("Time Step")
    plt.ylabel("CPU Demand")
    plt.legend()
    plt.grid()

    plt.show()


# =========================
# INDIVIDUAL POLICY PLOTS
# =========================

plot_policy("Rolling", "capacity")
plot_policy("Target Tracking", "tt_capacity")
plot_policy("OR-Tools", "ort_capacity")
plot_policy("Static", "static_capacity")

# =========================
# COMBINED GRAPH
# =========================
plt.figure(figsize=(12, 6))

plt.plot(df["cpu_demand"], label="Demand", linewidth=2)
plt.plot(df["capacity"], label="Rolling")
plt.plot(df["tt_capacity"], label="Target Tracking")
plt.plot(df["ort_capacity"], label="OR-Tools")
plt.plot(df["static_capacity"], label="Static", linestyle="--")

plt.title("All Policies Comparison")
plt.xlabel("Time Step")
plt.ylabel("CPU Demand")
plt.legend()
plt.grid()

plt.show()