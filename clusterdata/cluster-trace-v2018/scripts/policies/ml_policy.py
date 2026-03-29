import numpy as np
from sklearn.linear_model import LinearRegression


# ==============================
# TRAIN ML MODEL
# ==============================
def train_ml_model(df, window=10):
    """
    Train regression model on past window values → next value.
    TRAIN DATA ONLY.
    """

    X = []
    y = []

    data = df["task_arrivals"].values

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
def run_ml_policy(df, model, config, window=10):

    data = df["task_arrivals"].values

    instances = []
    prev_instances = 0

    for t in range(window, len(data)):

        # ==============================
        # ML prediction
        # ==============================
        window_data = data[t - window:t]
        pred = model.predict([window_data])[0]

        current = data[t]

        # 🔥 IMPROVED PREDICTION
        pred = 0.7 * pred + 0.3 * current
        pred = pred * 1.1   # slightly aggressive to reduce SLA

        # ==============================
        # Convert to instances
        # ==============================
        ml_instances = int(np.ceil(
            pred / config["capacity_per_instance"]
        ))

        # Apply bounds
        ml_instances = max(0, min(ml_instances, config["max_instances"]))

        # Smooth scaling (reduce jitter)
        ml_instances = int(0.6 * prev_instances + 0.4 * ml_instances)

        instances.append(ml_instances)
        prev_instances = ml_instances

    return np.array(instances)


# ==============================
# HYBRID POLICY
# ==============================
def run_hybrid_policy(df, model, config, window=10):

    data = df["task_arrivals"].values

    instances = []
    prev_instances = 0

    for t in range(window, len(data)):

        # ==============================
        # ML prediction
        # ==============================
        window_data = data[t - window:t]
        pred = model.predict([window_data])[0]

        current = data[t]

        pred = 0.7 * pred + 0.3 * current
        pred = pred * 1.1

        ml_instances = int(np.ceil(
            pred / config["capacity_per_instance"]
        ))

        # ==============================
        # Reactive fallback
        # ==============================
        reactive_instances = int(np.ceil(
            current / config["capacity_per_instance"]
        ))

        # ==============================
        # 🔥 TRUE HYBRID LOGIC
        # ==============================

        # ==============================
        # TRUE HYBRID LOGIC (FIXED)
        # ==============================

        # Safety: if ML underestimates → use reactive
        if ml_instances < reactive_instances:
            inst = reactive_instances

        # If ML overshoots too much → scale it down
        elif ml_instances > reactive_instances * 1.2:
            inst = int(0.7 * ml_instances)

        # Otherwise blend
        else:
            inst = int(0.6 * ml_instances + 0.4 * reactive_instances)

        # Apply bounds
        inst = max(0, min(inst, config["max_instances"]))

        # Smooth scaling
        inst = int(0.6 * prev_instances + 0.4 * inst)

        instances.append(inst)
        prev_instances = inst

    return np.array(instances)