from ortools.sat.python import cp_model
import os
import pandas as pd


def run_ortools_policy(demand, instance_cost, C, lambda_penalty, x_max, delta_max):

    model = cp_model.CpModel()
    T = len(demand)

    x = []
    under = []

    # Decision variables
    for t in range(T):
        x_t = model.NewIntVar(0, x_max, f'x_{t}')
        x.append(x_t)

        u = model.NewIntVar(0, 10**9, f'under_{t}')
        under.append(u)

        # Under-provision constraint
        model.Add(u >= demand[t] - x_t * C)

    # Smoothness constraint
    for t in range(1, T):
        model.Add(x[t] - x[t-1] <= delta_max)
        model.Add(x[t-1] - x[t] <= delta_max)

    # Objective
    model.Minimize(
        sum(instance_cost * x[t] + lambda_penalty * under[t] for t in range(T))
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60

    status = solver.Solve(model)

    if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
        print("No feasible solution found.")
        return None

    allocation = []
    total_under = 0
    total_instance_cost = 0
    sla_violations = 0

    for t in range(T):
        xt = solver.Value(x[t])
        ut = solver.Value(under[t])

        allocation.append(xt)
        total_under += ut
        total_instance_cost += xt * instance_cost

        if ut > 0:
            sla_violations += 1

    sla_violation_rate = sla_violations / T

    return (
        allocation,
        total_under,
        total_instance_cost,
        sla_violation_rate,
        solver.ObjectiveValue()
    )


if __name__ == "__main__":

    # Load sample trace
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "data", "sample_trace.csv")

    df = pd.read_csv(data_path)

    print("Columns:", df.columns)
    print(df.head())

    # Use correct demand column
    demand = df["task_arrivals"].values.tolist()[:300]  # limit for performance

    # Parameters (adjustable)
    instance_cost = 10
    C = 100
    lambda_penalty = 50
    x_max = 250
    delta_max = 10

    results = run_ortools_policy(
        demand=demand,
        instance_cost=instance_cost,
        C=C,
        lambda_penalty=lambda_penalty,
        x_max=x_max,
        delta_max=delta_max
    )

    if results:
        allocation, total_under, total_instance_cost, sla_rate, total_obj = results

        print("\n===== OR-Tools Offline Optimal Policy =====")
        print("Total instance cost:", total_instance_cost)
        print("Total unmet demand:", total_under)
        print("SLA violation rate:", sla_rate)
        print("Total objective value:", total_obj)
        print("Max demand:", max(demand))