import numpy as np
from sklearn.linear_model import LinearRegression


# ==============================
# TRAIN ML MODEL
# ==============================
def train_ml_model(df, config):
    """
    Train a linear regression model using the previous window values
    to predict the next workload demand value.
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

    Uses previous workload windows to predict demand, then converts
    predicted demand into instance count.
    """

    if window is None:
        window = config["window"]

    demand_col = config.get("demand_col", "task_arrivals")
    data = df[demand_col].values

    instances = []
    prev_instances = 0

    for t in range(window, len(data)):

        # Use only past demand values
        window_data = data[t - window:t]

        # Predict current/next demand from past window
        pred = model.predict([window_data])[0]

        # Slight safety buffer to reduce under-provisioning
        pred = pred * config.get("ml_safety_buffer", 1.1)

        # Convert predicted demand to instances
        desired_instances = int(np.ceil(
            pred / config["capacity_per_instance"]
        ))

        # Apply bounds
        desired_instances = max(
            0,
            min(desired_instances, config["max_instances"])
        )

        # Smooth scaling
        smoothing = config.get("ml_smoothing", 0.6)
        ml_instances = int(
            smoothing * prev_instances
            + (1 - smoothing) * desired_instances
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

    Combines ML prediction with a reactive safety check.
    The ML model predicts demand from previous windows.
    The reactive part checks current observed demand and prevents
    major under-provisioning.
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
        # Reactive guardrail
        # ==============================
        current_demand = data[t]

        reactive_instances = int(np.ceil(
            current_demand / config["capacity_per_instance"]
        ))

        # ==============================
        # Hybrid decision logic
        # ==============================

        # If ML underestimates, use reactive value for safety
        if ml_instances < reactive_instances:
            desired_instances = reactive_instances

        # If ML overshoots too much, reduce over-provisioning
        elif ml_instances > reactive_instances * hybrid_threshold:
            desired_instances = int(0.7 * ml_instances)

        # Otherwise blend ML and reactive decisions
        else:
            desired_instances = int(
                ml_weight * ml_instances
                + reactive_weight * reactive_instances
            )

        # Apply bounds
        desired_instances = max(
            0,
            min(desired_instances, config["max_instances"])
        )

        # Smooth scaling
        hybrid_instances = int(
            smoothing * prev_instances
            + (1 - smoothing) * desired_instances
        )

        instances.append(hybrid_instances)
        prev_instances = hybrid_instances

    return np.array(instances)