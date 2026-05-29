import numpy as np
from sklearn.linear_model import LinearRegression


# ==============================
# HELPER: MAX DELTA LIMIT
# ==============================
def apply_max_delta(prev_instances, desired_instances, max_delta):
    """
    Limit how much the policy can scale up or down in one time step.
    This makes ML and Hybrid obey the same scaling-rate constraint
    as the other policies.
    """

    if desired_instances > prev_instances:
        return min(prev_instances + max_delta, desired_instances)

    return max(prev_instances - max_delta, desired_instances)


# ==============================
# TRAIN ML MODEL
# ==============================
def train_ml_model(df, config):
    """
    Train a linear regression model using previous window values
    to predict future workload demand.
    """

    X = []
    y = []

    demand_col = config.get("demand_col", "task_arrivals")
    window = config["window"]

    data = df[demand_col].values

    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(data[i])

    X = np.array(X)
    y = np.array(y)

    model = LinearRegression()
    model.fit(X, y)

    return model


# ==============================
# PURE ML POLICY
# ==============================
def run_ml_policy(df, model, config, window=None):
    """
    Pure ML autoscaling policy.

    Uses only past workload values to predict demand.
    It does not use current demand directly.
    """

    if window is None:
        window = config["window"]

    demand_col = config.get("demand_col", "task_arrivals")
    data = df[demand_col].values

    instances = []
    prev_instances = 0

    for t in range(window, len(data)):

        # Predict using only past values
        window_data = data[t - window:t]
        pred = model.predict([window_data])[0]

        # Safety buffer
        pred = pred * config.get("ml_safety_buffer", 1.1)

        desired_instances = int(np.ceil(
            pred / config["capacity_per_instance"]
        ))

        desired_instances = max(
            0,
            min(desired_instances, config["max_instances"])
        )

        # Smooth the desired value
        smoothing = config.get("ml_smoothing", 0.6)

        smoothed_instances = int(
            smoothing * prev_instances
            + (1 - smoothing) * desired_instances
        )

        # Enforce shared max_delta scaling constraint
        ml_instances = apply_max_delta(
            prev_instances,
            smoothed_instances,
            config["max_delta"]
        )

        instances.append(ml_instances)
        prev_instances = ml_instances

    return np.array(instances)


# ==============================
# HYBRID POLICY
# ==============================
def run_hybrid_policy(df, model, config, window=None):
    """
    Hybrid autoscaling policy.

    Combines ML prediction with a reactive safety guardrail.
    The reactive part can correct ML underestimation, but the final
    scaling action still obeys max_delta for fairness.
    """

    if window is None:
        window = config["window"]

    demand_col = config.get("demand_col", "task_arrivals")
    data = df[demand_col].values

    instances = []
    prev_instances = 0

    hybrid_threshold = config.get("hybrid_threshold", 1.2)
    ml_weight = config.get("hybrid_ml_weight", 0.6)
    reactive_weight = config.get("hybrid_reactive_weight", 0.4)
    smoothing = config.get("hybrid_smoothing", 0.6)

    for t in range(window, len(data)):

        # ==============================
        # ML prediction from past values
        # ==============================
        window_data = data[t - window:t]
        pred = model.predict([window_data])[0]

        pred = pred * config.get("ml_safety_buffer", 1.1)

        ml_instances = int(np.ceil(
            pred / config["capacity_per_instance"]
        ))

        # ==============================
        # Reactive guardrail using current observed demand
        # ==============================
        current_demand = data[t]

        reactive_instances = int(np.ceil(
            current_demand / config["capacity_per_instance"]
        ))

        # ==============================
        # Hybrid decision logic
        # ==============================

        # If ML underestimates, use reactive demand as the target
        if ml_instances < reactive_instances:
            desired_instances = reactive_instances

        # If ML overshoots too much, reduce over-provisioning
        elif ml_instances > reactive_instances * hybrid_threshold:
            desired_instances = int(0.7 * ml_instances)

        # Otherwise blend both decisions
        else:
            desired_instances = int(
                ml_weight * ml_instances
                + reactive_weight * reactive_instances
            )

        desired_instances = max(
            0,
            min(desired_instances, config["max_instances"])
        )

        # Smooth the desired value
        smoothed_instances = int(
            smoothing * prev_instances
            + (1 - smoothing) * desired_instances
        )

        # Try to preserve the reactive safety target, but still obey max_delta
        if ml_instances < reactive_instances:
            smoothed_instances = max(smoothed_instances, reactive_instances)

        # Enforce shared max_delta scaling constraint
        hybrid_instances = apply_max_delta(
            prev_instances,
            smoothed_instances,
            config["max_delta"]
        )

        instances.append(hybrid_instances)
        prev_instances = hybrid_instances

    return np.array(instances)