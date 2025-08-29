import os
import time
import json
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from itertools import permutations
from scipy.optimize import minimize, differential_evolution
import pandas as pd # <-- IMPORT ADDED HERE

from main import (
    PauliEvolutionSimulator,
    create_tsp_instance,
    get_hamiltonian_pauli_strings,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- All helper functions remain the same ---
def valid_bitstrings(num_cities):
    outs = []
    for tour in permutations(range(num_cities)):
        mat = np.zeros((num_cities, num_cities), dtype=int)
        for pos, city in enumerate(tour): mat[city, pos] = 1
        outs.append("".join(map(str, mat.flatten())))
    return outs

def feasibility_metrics(state_vec, num_cities):
    n_qubits = num_cities**2
    probs = torch.abs(state_vec)**2
    V = valid_bitstrings(num_cities)
    idxs = [int(b, 2) for b in V]
    p_valid = float(torch.sum(probs[idxs]).item())
    return p_valid

def list_valid_tours_with_probs(state_vec, num_cities):
    n_qubits = num_cities**2; probs = torch.abs(state_vec)**2; rows = []
    for b in valid_bitstrings(num_cities):
        idx = int(b, 2); prob = probs[idx].item()
        if prob > 1e-12:
            mat = np.array(list(b), dtype=int).reshape(num_cities, num_cities)
            tour = list(np.argmax(mat, axis=1)); rows.append((tour, prob))
    rows.sort(key=lambda x: -x[1]); return rows

def get_tour_distance(tour, dist_matrix):
    distance = 0.0
    for i in range(len(tour)): distance += dist_matrix[tour[i], tour[(i + 1) % len(tour)]]
    return float(distance)

def save_artifacts(output_dir, file_name, content, is_json=False):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, file_name)
    with open(path, "w") as f:
        if is_json: json.dump(content, f, indent=2)
        else: f.write(content)

def solve_tsp_classical(dist_matrix):
    print("\n--- Running Classical Brute-Force Solver ---")
    t0 = time.time()
    num_cities = dist_matrix.shape[0]
    all_tours = list(permutations(range(num_cities)))
    best_tour = None
    optimal_distance = float('inf')
    for tour in all_tours:
        dist = get_tour_distance(tour, dist_matrix)
        if dist < optimal_distance:
            optimal_distance = dist
            best_tour = tour
    runtime = time.time() - t0
    print(f"Classical solver finished in {runtime:.6f} seconds.")
    print(f"Optimal tour: {list(best_tour)} | Optimal distance: {optimal_distance:.6f}")
    return list(best_tour), optimal_distance, runtime

def main():
    print("--- Starting Definitive QAOA Experiment ---")

    NUM_CITIES = int(os.environ.get("NUM_CITIES", 4))
    P_VALUES   = os.environ.get("P_VALUES", "1,2,3")
    OPTIMIZER  = os.environ.get("OPTIMIZER", "COBYLA")
    NUM_STARTS = int(os.environ.get("NUM_STARTS", 5))
    MAXITER    = int(os.environ.get("MAXITER", 150))
    SEED       = int(os.environ.get("SEED", 42))

    rng = np.random.default_rng(SEED)
    p_list = [int(x.strip()) for x in P_VALUES.split(",") if x.strip()]
    output_dir = os.environ.get("AIP_MODEL_DIR", "outputs")
    
    print(f"Config: Cities={NUM_CITIES}, P={p_list}, Optimizer={OPTIMIZER}, Starts={NUM_STARTS}, MaxIter={MAXITER}, Seed={SEED}")

    dist_matrix = create_tsp_instance(NUM_CITIES)
    optimal_tour, optimal_distance, classical_runtime = solve_tsp_classical(dist_matrix)

    pauli_terms, n_qubits = get_hamiltonian_pauli_strings(dist_matrix)
    sim = PauliEvolutionSimulator(pauli_terms, n_qubits)
    initial_state = torch.ones(2**n_qubits, dtype=torch.cfloat, device=device) / math.sqrt(2**n_qubits)
    
    metrics_rows = []
    for P_LAYERS in p_list:
        print(f"\n========== QAOA depth p={P_LAYERS} ==========")
        t0 = time.time()
        best_fun, best_params, eval_counter = None, None, {"count": 0}

        def objective_function(params):
            eval_counter["count"] += 1
            final_state = sim.run_qaoa_step(initial_state, params, P_LAYERS)
            cost = sim.get_expectation_value_for_state(final_state)
            return cost
        
        for s in range(NUM_STARTS):
            print(f"\n--- Start {s+1}/{NUM_STARTS} (p={P_LAYERS}, seed={SEED}) ---")
            init = rng.random(2 * P_LAYERS) * np.pi
            
            if OPTIMIZER.upper() == 'DIFFERENTIAL_EVOLUTION':
                bounds = [(0, 2*np.pi)] * (2 * P_LAYERS)
                result = differential_evolution(objective_function, bounds, maxiter=MAXITER, seed=SEED)
            else:
                result = minimize(objective_function, init, method="COBYLA", options={"maxiter": MAXITER})

            if (best_fun is None) or (result.fun < best_fun):
                best_fun, best_params = result.fun, result.x
                print(f"  New best cost (p={P_LAYERS}): {best_fun:.6f}")

        final_state_optimized = sim.run_qaoa_step(initial_state, best_params, P_LAYERS)
        p_valid = feasibility_metrics(final_state_optimized, NUM_CITIES)
        valid_results = list_valid_tours_with_probs(final_state_optimized, NUM_CITIES)

        best_tour, best_prob, best_distance, approximation_ratio = None, 0.0, float("nan"), float("inf")
        if valid_results:
            best_tour, best_prob = valid_results[0]
            best_distance = get_tour_distance(best_tour, dist_matrix)
            approximation_ratio = best_distance / optimal_distance
        
        qaoa_runtime = time.time() - t0
        
        metrics_rows.append([P_LAYERS, approximation_ratio, p_valid, best_prob, qaoa_runtime, classical_runtime])

    df = pd.DataFrame(metrics_rows, columns=["p", "approximation_ratio", "p_valid", "best_prob", "qaoa_runtime_sec", "classical_runtime_sec"])
    save_artifacts(output_dir, "final_metrics.csv", df.to_csv(index=False))
    
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(); plt.plot(df['p'], df['approximation_ratio'], marker="o"); plt.axhline(y=1.0, color='r', linestyle='--'); plt.xlabel("QAOA depth p"); plt.ylabel("Approximation Ratio"); plt.title("Solution Quality vs Circuit Depth"); plt.savefig(os.path.join(output_dir, "approx_ratio_vs_p.png")); plt.close()
    plt.figure(); plt.plot(df['p'], df['qaoa_runtime_sec'], marker="o", label="QAOA Optimization"); plt.axhline(y=df['classical_runtime_sec'].iloc[0], color='r', linestyle='--', label="Classical Exact Solver"); plt.xlabel("QAOA depth p"); plt.ylabel("Runtime (sec)"); plt.title("Runtime Comparison: QAOA vs Classical"); plt.legend(); plt.savefig(os.path.join(output_dir, "runtime_comparison.png")); plt.close()

    print(f"\n✅ All artifacts (final_metrics.csv, plots) saved to GCS at: {output_dir}")

if __name__ == "__main__":
    main()
