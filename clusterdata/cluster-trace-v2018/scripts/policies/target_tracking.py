import numpy as np

def run_target_tracking_policy(df, config):

    df = df.copy()

    TARGET_UTIL = 0.7
    COOLDOWN = 0
    DELAY = 1

    instances = []
    last_scale_time = -COOLDOWN
    pending = []  # (time_to_apply, new_instances)

    current_instances = 0

    for t in range(len(df)):

        demand_t = df.loc[t, "task_arrivals"]

        # --- Apply delayed scaling actions ---
        for change in pending[:]:
            apply_time, new_x = change
            if t >= apply_time:
                current_instances = new_x
                pending.remove(change)

        # --- Compute desired instances ---
        desired = int(np.ceil(
            demand_t / (TARGET_UTIL * config["capacity_per_instance"])
        ))

        desired = min(desired, config["max_instances"])

        # --- Cooldown + delta constraint ---
        if t - last_scale_time >= COOLDOWN:

            if desired > current_instances:
                new_x = min(current_instances + config["max_delta"], desired)
            else:
                new_x = max(current_instances - config["max_delta"], desired)

            if new_x != current_instances:
                pending.append((t + DELAY, new_x))
                last_scale_time = t

        instances.append(current_instances)

    df["instances"] = instances

    # ==============================
    # METRICS
    # ==============================

    df["capacity"] = df["instances"] * config["capacity_per_instance"]

    df["under"] = np.maximum(
        0, df["cpu_demand"] - df["capacity"]
    )

    df["over"] = np.maximum(
        0, df["capacity"] - df["cpu_demand"]
    )

    df["cost"] = (
        df["instances"] * config["instance_cost"]
        + df["under"] * config["under_penalty"]
    )

    stability = df["instances"].diff().abs().sum()

    return {
    "metrics": {
        "total_cost": df["cost"].sum(),
        "total_under": df["under"].sum(),
        "total_over": df["over"].sum(),
        "sla_violation_rate":
            (df["under"] > 0).sum() / len(df) * 100,
        "stability": stability
    },
    "instances": df["instances"].values
}