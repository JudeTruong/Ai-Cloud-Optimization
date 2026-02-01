import pandas as pd
import numpy as np

SEED = 42 # we can change for diff data but when we hardcode we can reproduce 
SAMPLE_WINDOWS = 1_000  # number of 5-min windows we are selecting from our clean data

np.random.seed(SEED) #makes a reproduible seed so np.random is always the same based on seed

# Load cleaned arrival data
df = pd.read_csv(
    "clusterdata/cluster-trace-v2018/data/alibaba_task_arrivals_5min.csv"
)

# Ensure temporal order
df = df.sort_values("time").reset_index(drop=True)# sorts data

# Reproducible time-based sampling
start_idx = np.random.randint(0, len(df) - SAMPLE_WINDOWS) #random start based on seed
sample_df = df.iloc[start_idx : start_idx + SAMPLE_WINDOWS] # takes the slice of rows as big as sample windows

# Freeze dataset
sample_df.to_csv(
    "clusterdata/cluster-trace-v2018/data/sample_trace.csv",
    index=False
)
print()
print()
print()
print("----------------Output for Sampling Trace .py ------------------")
print("Final sampled dataset:", sample_df.shape)
print("Time range:", sample_df["time"].iloc[0], "→", sample_df["time"].iloc[-1])
print("----------------end of Output for Sampling Trace .py ------------------")
print()
print()
print()
