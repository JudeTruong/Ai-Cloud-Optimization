import pandas as pd

WINDOW = 3  # number of past windows to use

# Load frozen dataset made by sample_trace testing
df = pd.read_csv(
    "clusterdata/cluster-trace-v2018/data/sample_trace.csv"
)

# Ensure order of taken windows
df = df.sort_values("time").reset_index(drop=True)

# Rolling average predictor, take only the 5 windows and use it to make the next one
df["prediction"] = (
    df["task_arrivals"]
    .rolling(window=WINDOW) #avg of last t entries
    .mean()
    .shift(1)  # predict next window
)
FAILURE_THRESHOLD = 1.25

# Under-provisioning failures
df["under_provisioned"] = df["task_arrivals"] > df["prediction"]

# Drop rows where prediction is undefined
df = df.dropna().reset_index(drop=True)


print(df[["time", "task_arrivals", "prediction"]].head(10)) #only show the predition for the first 10 
total_windows = len(df)
failures = df["under_provisioned"].sum()
failure_rate = failures / total_windows * 100

print("=== Under-Provisioning Summary ===")
print(f"Total windows: {total_windows} this is beacuse we cant use the first 10 for rolling averages")
print(f"Under-provisioned windows: {failures}: total amount of tasks not provisoned properly")
print(f"Failure rate: {failure_rate:.2f}%")
