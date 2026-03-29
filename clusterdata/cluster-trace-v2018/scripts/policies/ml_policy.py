import numpy as np
from sklearn.linear_model import LinearRegression

#training it
def train_ml_model(df, window):

    data = df["task_arrivals"].values

    X = []
    y = []

    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])

    X = np.array(X)
    y = np.array(y)

    model = LinearRegression()
    model.fit(X, y)

    return model

#doing it 
def run_ml_policy(df, model, config, window):

    data = df["task_arrivals"].values

    instances = []

    for t in range(window, len(data)):

        window_data = data[t-window:t]
        pred = model.predict([window_data])[0]
        pred = pred * 0.8

        ml_instances = int(np.ceil(pred / config["capacity_per_instance"]))

        ml_instances = max(0, min(ml_instances, config["max_instances"]))

        instances.append(ml_instances)

    return np.array(instances)


#hybrid with safeguards
def run_hybrid_policy(df, model, config, window):

    data = df["task_arrivals"].values

    instances = []

    for t in range(window, len(data)):

        # ML prediction
        window_data = data[t-window:t]
        pred = model.predict([window_data])[0]
        pred = pred * 0.8

        ml_instances = int(np.ceil(pred / config["capacity_per_instance"]))

        # Reactive baseline (current demand)
        current_demand = data[t]
        reactive_instances = int(np.ceil(
            current_demand / config["capacity_per_instance"]
        ))

        # SOFT HYBRID (NOT max)
        inst = int(0.7 * ml_instances + 0.3 * reactive_instances)

        inst = max(0, min(inst, config["max_instances"]))

        instances.append(inst)

    return np.array(instances)