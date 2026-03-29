import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from run_experiment import run_experiment

# =========================
# CONFIG
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

# Demand
demand = df["cpu_demand"]

# =========================
# POLICY SELECTOR
# =========================

mapping = {
    "rolling": ("Rolling", "rolling_capacity"),
    "tt": ("Target Tracking", "tt_capacity"),
    "ml": ("ML", "ml_capacity"),
    "hybrid": ("Hybrid", "hybrid_capacity"),
    "ort": ("OR-Tools", "ort_capacity"),
    "static": ("Static", "static_capacity"),
}

# =========================
# SIDE-BY-SIDE POLICIES
# =========================

LEFT_POLICY = "static"
RIGHT_POLICY = "hybrid"

name1, col1 = mapping[LEFT_POLICY]
name2, col2 = mapping[RIGHT_POLICY]

# =========================
# FIGURE SETUP
# =========================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

y_max = max(demand) * 1.1  # fixed scale for smoothness

# =========================
# ANIMATION FUNCTION
# =========================

def update(frame):
    ax1.clear()
    ax2.clear()

    # LEFT POLICY
    ax1.plot(demand[:frame], linewidth=2, label="Demand")
    ax1.plot(df[col1][:frame], label=name1)
    ax1.fill_between(
        range(frame),
        demand[:frame],
        df[col1][:frame],
        where=(demand[:frame] > df[col1][:frame]),
        alpha=0.3,
    )
    ax1.set_title(name1)
    ax1.set_xlim(0, len(df))
    ax1.set_ylim(0, y_max)
    ax1.legend()
    ax1.grid()

    # RIGHT POLICY
    ax2.plot(demand[:frame], linewidth=2, label="Demand")
    ax2.plot(df[col2][:frame], label=name2)
    ax2.fill_between(
        range(frame),
        demand[:frame],
        df[col2][:frame],
        where=(demand[:frame] > df[col2][:frame]),
        alpha=0.3,
    )
    ax2.set_title(name2)
    ax2.set_xlim(0, len(df))
    ax2.set_ylim(0, y_max)
    ax2.legend()
    ax2.grid()

# =========================
# ANIMATION SETTINGS
# =========================

ani = FuncAnimation(
    fig,
    update,
    frames=range(0, len(df), 2),  # smooth playback
    interval=40,                  # speed
    blit=False
)

plt.show()

# =========================
# OPTIONAL: SAVE GIF
# =========================
# ani.save("comparison.gif", writer="pillow", fps=30)