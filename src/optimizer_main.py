import os
import time
import json
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from itertools import permutations
from scipy.optimize import minimize

from main import (
    PauliEvolutionSimulator,
    create_tsp_instance,
    get_hamiltonian_pauli_strings,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def valid_bitstrings(num_cities):
    outs = []
    for tour in permutations(range(num_cities)):
        mat = np.zeros((num_cities, num_cities), dtype=int)
        for pos, city in enumerate(tour):
            mat[city, pos] = 1
        outs.append("".join(map(str, mat.flatten())))
    return outs

def feasibility_metrics(state_vec, num_cities, n_qubits):
    probs = torch.abs(state_vec)**2
    V = valid_bitstrings(num_cities)
    idxs = [int(b, 2) for b in V]
    p_valid = float(torch.sum(probs[idxs]).item())
    p_invalid = 1.0 - p_valid
    return p_valid, p_invalid

def list_valid_tours_with_probs(state_vec, num_cities):
    n_qubits = num_cities * num_cities
    probs = torch.abs(state_vec)**2
    rows = []
    for b in valid_bitstrings(num_cities):
        idx = int(b, 2)
        prob = probs[idx].item()
        if prob > 1e-12:
            mat = np.array(list(b), dtype=int).reshape(num_cities, num_cities)
            tour = list(np.argmax(mat, axis=1))
            rows.append((tour, prob))
    rows.sort(key=lambda x: -x[1])
    return rows

def get_tour_distance(tour, dist_matrix):
    distance = 0.0
    for i in range(len(tour)):
        distance += dist_matrix[tour[i], tour[(i + 1) % len(tour)]]
    return float(distance)

def save_csv(path, header, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        if header:
            f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def main():
    print("--- Starting Multi-Depth QAOA Optimization for TSP ---")

    # Vertex AI provides a GCS path for saving artifacts
    output_dir = os.environ.get("AIP_MODEL_DIR", "outputs")
    print(f"Artifacts will be saved to: {output_dir}")

    NUM_CITIES = int(os.environ.get("NUM_CITIES", 4))
    P_VALUES   = os.environ.get("P_VALUES", "1,2,3")
    NUM_STARTS = int(os.environ.get("NUM_STARTS", 5))
    MAXITER    = int(os.environ.get("MAXITER", 200))
    SEED       = int(os.environ.get("SEED", 42))

    p_list = [int(x.strip()) for x in P_VALUES.split(",") if x.strip()]
    print(f"Cities: {NUM_CITIES}, P list: {p_list}, starts: {NUM_STARTS}, maxiter: {MAXITER}, seed: {SEED}")

    dist_matrix = create_tsp_instance(NUM_CITIES)
    pauli_terms, n_qubits = get_hamiltonian_pauli_strings(dist_matrix)
    sim = PauliEvolutionSimulator(pauli_terms, n_qubits)

    dim = 2**n_qubits
    initial_state = torch.ones(dim, dtype=torch.cfloat, device=device) / math.sqrt(dim)

    all_tours = list(permutations(range(NUM_CITIES)))
    optimal_distance = min([get_tour_distance(tour, dist_matrix) for tour in all_tours])
    print(f"Optimal (brute-force) tour distance: {optimal_distance:.6f}")

    rng = np.random.default_rng(SEED)
    metrics_rows = []
    best_params_by_p = {}

    for P_LAYERS in p_list:
        print(f"\n========== QAOA depth p={P_LAYERS} ==========")
        t0 = time.time()
        best_fun, best_params, eval_counter = None, None, {"count": 0}

        def objective_function(params):
            eval_counter["count"] += 1
            final_state = sim.run_qaoa_step(initial_state, params, P_LAYERS)
            cost = sim.get_expectation_value_for_state(final_state)
            print(f"    [p={P_LAYERS}] eval={eval_counter['count']:03d} cost={cost:.6f}")
            return cost

        for s in range(NUM_STARTS):
            print(f"\n--- Start {s+1}/{NUM_STARTS} (p={P_LAYERS}) ---")
            init = rng.random(2 * P_LAYERS) * np.pi
            result = minimize(objective_function, init, method="COBYLA", options={"maxiter": MAXITER})
            if (best_fun is None) or (result.fun < best_fun):
                best_fun, best_params = result.fun, result.x
                print(f"  New best cost (p={P_LAYERS}): {best_fun:.6f}")

        final_state_optimized = sim.run_qaoa_step(initial_state, best_params, P_LAYERS)
        p_valid, p_invalid = feasibility_metrics(final_state_optimized, NUM_CITIES, n_qubits)
        valid_results = list_valid_tours_with_probs(final_state_optimized, NUM_CITIES)

        if valid_results:
            best_tour, best_prob = valid_results[0]
            best_distance = get_tour_distance(best_tour, dist_matrix)
            approximation_ratio = best_distance / optimal_distance
        else:
            best_tour, best_prob, best_distance, approximation_ratio = None, 0.0, float("nan"), float("inf")

        runtime = time.time() - t0
        
        vt_rows = [[ " ".join(map(str, tour)), prob, get_tour_distance(tour, dist_matrix) ] for tour, prob in valid_results]
        save_csv(os.path.join(output_dir, f"qaoa_valid_tours_p{P_LAYERS}.csv"), header=["tour", "probability", "distance"], rows=vt_rows)
        best_params_by_p[str(P_LAYERS)] = {"params": best_params.tolist(), "best_cost": best_fun, "evals": eval_counter["count"]}
        metrics_rows.append([P_LAYERS, best_prob, best_distance, optimal_distance, approximation_ratio, p_valid, p_invalid, runtime, NUM_STARTS, MAXITER, eval_counter["count"]])

        if valid_results:
            plt.figure(); plt.bar(range(len(valid_results)), [p for _, p in valid_results]); plt.xlabel("Valid tours (index)"); plt.ylabel("Probability"); plt.title(f"QAOA valid tour probability distribution (p={P_LAYERS})"); plt.savefig(os.path.join(output_dir, f"qaoa_tour_hist_p{P_LAYERS}.png")); plt.close()

    save_csv(os.path.join(output_dir, "metrics_by_p.csv"), header=["p", "best_prob", "best_distance", "optimal_distance", "approximation_ratio", "p_valid", "p_invalid", "runtime_sec", "num_starts", "maxiter", "objective_evals"], rows=metrics_rows)
    save_json(os.path.join(output_dir, "best_params_by_p.json"), best_params_by_p)

    ps, approx, p_valids, runtimes = ([r[0] for r in metrics_rows], [r[4] for r in metrics_rows], [r[5] for r in metrics_rows], [r[7] for r in metrics_rows])
    
    plt.figure(); plt.plot(ps, approx, marker="o"); plt.xlabel("QAOA depth p"); plt.ylabel("Approximation ratio"); plt.title("Approximation ratio vs QAOA depth"); plt.savefig(os.path.join(output_dir, "approx_ratio_vs_p.png")); plt.close()
    plt.figure(); plt.plot(ps, p_valids, marker="o"); plt.xlabel("QAOA depth p"); plt.ylabel("Feasible probability mass"); plt.title("Feasibility mass vs QAOA depth"); plt.savefig(os.path.join(output_dir, "p_valid_vs_p.png")); plt.close()
    
    print(f"\nAll artifacts written to GCS directory: {output_dir}")

if __name__ == "__main__":
    main()
