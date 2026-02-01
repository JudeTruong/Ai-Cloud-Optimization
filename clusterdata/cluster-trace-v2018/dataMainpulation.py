# --------------------------------------------------
# CP395 – AI-Driven Cloud Optimization
# Week 03: Data Ingestion & Cleaning
# Dataset: Alibaba Cluster Trace 2018
# --------------------------------------------------

import pandas as pd
import os
import numpy as np

SEED = 42
np.random.seed(SEED)

# --------------------------------------------------
# Step 0: Resolve file paths safely
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "batch_task.csv")

# --------------------------------------------------
# Step 1: Load raw data (INGESTION)
# --------------------------------------------------
# The Alibaba batch_task CSV has NO header row.
# Column meanings are defined in schema.txt.
#
# From the schema and inspection:
# Column index 5 = task submission time
#
# We:
# - Tell pandas there is no header (header=None)
# - Select column 5 only (usecols=[5])
# - Assign a meaningful column name ourselves

df = pd.read_csv(
    DATA_PATH,
    header=None,
    usecols=[5],
    names=["submit_time"]
)


print("Raw submission times:")
print(df.head())

# --------------------------------------------------
# Step 2: Timestamp normalization (CLEANING)
# --------------------------------------------------
# submit_time is stored as seconds since trace start.
# Convert it into a datetime object so we can
# perform time-based aggregation.

df["time"] = pd.to_datetime(df["submit_time"], unit="s")

# Set the timestamp as the index
df = df.set_index("time")
df = df.sort_index()
# --------------------------------------------------
# Step 3: Create a countable column
# --------------------------------------------------
# After setting the index, we need something to
# count or sum when aggregating.
#
# Each task represents ONE arrival.

df["one"] = 1

# --------------------------------------------------
# Step 4: Temporal aggregation (CORE CLEANING STEP)
# --------------------------------------------------
# Autoscaling operates on fixed control intervals,
# not per-task events.
#
# We aggregate task arrivals into 5-minute windows
# by summing the dummy "one" column.

workload = df.resample("5min")["one"].sum()

# Convert Series to DataFrame and name the column
workload = workload.to_frame(name="task_arrivals")

print("\nAggregated workload (5-minute windows):")
print(workload.head())

# --------------------------------------------------
# Step 5: Handle missing windows
# --------------------------------------------------
# Windows with no arrivals should be treated as
# zero load, not missing data.

workload = workload.fillna(0)

# --------------------------------------------------
# Step 6: Save cleaned intermediate dataset
# --------------------------------------------------
# This file represents the cleaned workload and
# will be reused for EDA and modeling.
# Raw data remains untouched.

OUTPUT_PATH = os.path.join(BASE_DIR, "data", "alibaba_task_arrivals_5min.csv")
workload.to_csv(OUTPUT_PATH)




# Remove initial warm-up window
workload = workload.iloc[1:]
import matplotlib.pyplot as plt

workload.plot(
    figsize=(10,4),
    title="Alibaba Task Arrivals per 5-Minute Window"
)

plt.xlabel("Time")
plt.ylabel("Number of Tasks")
plt.tight_layout()
plt.show()


workload["task_arrivals"].hist(bins=50)

plt.title("Distribution of Task Arrivals")
plt.xlabel("Tasks per 5-Minute Window")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


rolling_mean = workload["task_arrivals"].rolling(12).mean()

rolling_mean.plot(
    figsize=(10,4),
    title="Rolling Mean of Task Arrivals (1-hour window)"
)

plt.xlabel("Time")
plt.ylabel("Average Tasks")
plt.tight_layout()
plt.show()
