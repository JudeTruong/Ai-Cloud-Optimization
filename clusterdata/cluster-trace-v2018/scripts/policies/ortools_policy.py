from ortools.sat.python import cp_model
import os
import pandas as pd


def run_ortools_policy(
    demand,
    instance_cost,
    C,
    lambda_penalty,
    x_max,
    delta_max,
    initial_instances=0
):
    """
    OR-Tools offline constrained optimization policy.

    This policy has full access to the demand trace, so it should be treated
    as an offline optimization benchmark, not a deployable online autoscaler.

    Objective:
        minimize total instance cost + under-provisioning penalty

    Constraints:
        - 0 <= x_t <= x_max
        - under_t >= demand_t - capacity_t
        - |x_t - x_{t-1}| <= delta_max
    """

    # CP-SAT requires integer objective coefficients.
    # Scaling allows decimal costs such as 0.02.
    COST_SCALE = 1000

    scaled_instance_cost = int(round(instance_cost * COST_SCALE))
    scaled_lambda_penalty = int(round(lambda_penalty * COST_SCALE))

    # Ensure demand is integer because CP-SAT uses integer variables.
    demand = [int(round(d)) for d in demand]

    model = cp_model.CpModel()
    T = len(demand)

    x = []
    under = []

    # ==============================
    # DECISION VARIABLES
    # ==============================
    for t in range(T):
        x_t = model.NewIntVar(0, x_max, f"x_{t}")
        x.append(x_t)

        # Under-provisioning cannot be negative.
        # Maximum possible under-provisioning is demand[t].
        u_t = model.NewIntVar(0, max(0, demand[t]), f"under_{t}")
        under.append(u_t)

        # unmet demand constraint:
        # under_t >= demand_t - allocated_capacity_t
        model.Add(u_t >= demand[t] - x_t * C)

    # ==============================
    # SCALING-RATE CONSTRAINT
    # ==============================

    # Limit first step from initial allocation
    if T > 0:
        model.Add(x[0] - initial_instances <= delta_max)
        model.Add(initial_instances - x[0] <= delta_max)

    # Limit changes between consecutive windows
    for t in range(1, T):
        model.Add(x[t] - x[t - 1] <= delta_max)
        model.Add(x[t - 1] - x[t] <= delta_max)

    # ==============================
    # OBJECTIVE FUNCTION
    # ==============================
    model.Minimize(
        sum(
            scaled_instance_cost * x[t]
            + scaled_lambda_penalty * under[t]
            for t in range(T)
        )
    )

    # ==============================
    # SOLVE
    # ==============================
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 180
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)

    print("OR-Tools solver status:", solver.StatusName(status))
    print("OR-Tools best objective:", solver.ObjectiveValue())
    print("OR-Tools best bound:", solver.BestObjectiveBound())

    if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
        print("No feasible solution found.")
        return None

    # ==============================
    # EXTRACT RESULTS
    # ==============================
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

    # Return real unscaled objective value
    total_objective_value = total_instance_cost + total_under * lambda_penalty

    return (
        allocation,
        total_under,
        total_instance_cost,
        sla_violation_rate,
        total_objective_value
    )


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "data", "sample_trace.csv")

    df = pd.read_csv(data_path)

    print("Columns:", df.columns)
    print(df.head())

    demand_col = "task_arrivals"
    demand = df[demand_col].values.tolist()[:300]

    instance_cost = 0.02
    C = 5
    lambda_penalty = 6
    x_max = 2000
    delta_max = 25

    results = run_ortools_policy(
        demand=demand,
        instance_cost=instance_cost,
        C=C,
        lambda_penalty=lambda_penalty,
        x_max=x_max,
        delta_max=delta_max,
        initial_instances=0
    )

    if results:
        allocation, total_under, total_instance_cost, sla_rate, total_obj = results

        stability = sum(
            abs(allocation[i] - allocation[i - 1])
            for i in range(1, len(allocation))
        )

        total_over = sum(
            max(0, allocation[t] * C - demand[t])
            for t in range(len(demand))
        )

        print("\n===== OR-Tools Offline Benchmark =====")
        print("Total instance cost:", total_instance_cost)
        print("Total unmet demand:", total_under)
        print("Total over-provisioning:", total_over)
        print("SLA violation rate:", sla_rate)
        print("Total objective value:", total_obj)
        print("Stability:", stability)
        print("Max demand:", max(demand))