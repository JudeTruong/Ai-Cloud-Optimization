from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

SCRIPT_DIR = ROOT / "clusterdata" / "cluster-trace-v2018" / "scripts"
DATA_FILE = ROOT / "clusterdata" / "cluster-trace-v2018" / "data" / "sample_trace.csv"

if not DATA_FILE.exists():
    print("Missing data file:")
    print(DATA_FILE)
    print()
    print("For now, add sample_trace.csv into:")
    print(DATA_FILE.parent)
    print()
    print("Later we will fix this by adding a reproducible data-generation script.")
    sys.exit(1)

print("Running autoscaling experiment...")
print(f"Script folder: {SCRIPT_DIR}")

subprocess.run(
    [sys.executable, "run_experiment.py"],
    cwd=SCRIPT_DIR,
    check=True
)

print("Experiment complete.")