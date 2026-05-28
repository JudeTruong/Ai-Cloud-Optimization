from metrics import compute_metrics


def run_static_policy(df, config):
    df = df.copy()

    static_instances = config["static_instances"]

    df["static_instances"] = static_instances

    return compute_metrics(df, "static_instances", config)