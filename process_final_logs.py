import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import json

def parse_log_file(log_file_path):
    """Parses a downloaded log CSV to extract structured metrics for the paper."""
    print(f"Reading log file: {log_file_path}...")
    try:
        df = pd.read_csv(log_file_path)
    except FileNotFoundError:
        print(f"Error: Log file not found at '{log_file_path}'")
        return None

    metrics_rows = []
    
    # Regex to find key metrics in the log messages
    p_regex = re.compile(r"========== QAOA depth p=(\d+) ==========")
    prob_regex = re.compile(r"Best valid tour \(p=\d+\): .* \| prob=([\d.e-]+) \|")
    dist_regex = re.compile(r"Best valid tour \(p=\d+\): .* \| dist=([\d.]+) \|")
    opt_dist_regex = re.compile(r"Optimal \(brute-force\) tour distance: ([\d.]+)")
    p_valid_regex = re.compile(r"Feasibility \(p=\d+\): p_valid=([\d.e-]+),")
    runtime_regex = re.compile(r"Distributed simulation complete in ([\d.]+) seconds.") # Assuming this is logged at the end of each p-run in a real scenario
    
    log_messages = ""
    if 'jsonPayload.message' in df.columns:
        log_messages = "\n".join(df['jsonPayload.message'].dropna().astype(str))
    elif 'textPayload' in df.columns:
        log_messages = "\n".join(df['textPayload'].dropna().astype(str))
    else:
        print("Error: Could not find a 'jsonPayload.message' or 'textPayload' column in the CSV.")
        return None

    optimal_distance_match = opt_dist_regex.search(log_messages)
    optimal_distance = float(optimal_distance_match.group(1)) if optimal_distance_match else float('nan')

    p_sections = p_regex.split(log_messages)

    if len(p_sections) > 1:
        for i in range(1, len(p_sections), 2):
            p_value = int(p_sections[i])
            section_log = p_sections[i+1]
            
            prob_match = prob_regex.search(section_log)
            dist_match = dist_regex.search(section_log)
            p_valid_match = p_valid_regex.search(section_log)
            # In this version, we assume the final runtime is the one we care about
            runtime_match = runtime_regex.search(log_messages)

            metrics_rows.append({
                "p_layers": p_value,
                "best_tour_probability": float(prob_match.group(1)) if prob_match else 0.0,
                "best_tour_distance": float(dist_match.group(1)) if dist_match else float('nan'),
                "optimal_distance": optimal_distance,
                "feasible_state_probability": float(p_valid_match.group(1)) if p_valid_match else 0.0,
                "total_runtime_seconds": float(runtime_match.group(1)) if runtime_match else float('nan'),
            })

    if not metrics_rows:
        print("Could not parse any metrics. Please check the log file content.")
        return None
        
    return pd.DataFrame(metrics_rows)

def generate_plots(df, output_dir="final_paper_artifacts"):
    """Generates and saves all plots from the final metrics DataFrame."""
    if df is None or df.empty:
        print("No data available to generate plots.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-darkgrid')

    # Plot 1: Feasibility vs. p
    plt.figure(figsize=(10, 6))
    plt.plot(df['p_layers'], df['feasible_state_probability'], marker="o", linestyle='-')
    plt.xlabel("QAOA Depth (p)")
    plt.ylabel("Feasible State Probability (p_valid)")
    plt.title("Feasibility vs. QAOA Depth")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "feasibility_vs_p.png"), dpi=300)
    plt.close()

    # Plot 2: Best Tour Probability vs. p
    plt.figure(figsize=(10, 6))
    plt.plot(df['p_layers'], df['best_tour_probability'], marker="o", linestyle='-')
    plt.xlabel("QAOA Depth (p)")
    plt.ylabel("Best Valid Tour Probability")
    plt.title("Best Tour Probability vs. QAOA Depth")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "best_prob_vs_p.png"), dpi=300)
    plt.close()

    # Plot 3: Best Tour Distance vs. p
    plt.figure(figsize=(10, 6))
    plt.plot(df['p_layers'], df['best_tour_distance'], marker="o", label="QAOA Found Distance")
    plt.axhline(y=df['optimal_distance'].iloc[0], color='r', linestyle='--', label='Optimal Distance')
    plt.xlabel("QAOA Depth (p)")
    plt.ylabel("Tour Distance")
    plt.title("Best Found Tour Distance vs. QAOA Depth")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "distance_vs_p.png"), dpi=300)
    plt.close()
    
    print(f"✅ All plots saved to '{output_dir}' directory.")

if __name__ == "__main__":
    log_file = "downloaded-logs-20250824-145717.csv"
    final_metrics_df = parse_log_file(log_file)
    
    if final_metrics_df is not None:
        output_dir = "final_paper_artifacts"
        # Save the final, processed CSV
        csv_path = os.path.join(output_dir, "final_paper_data.csv")
        os.makedirs(output_dir, exist_ok=True)
        final_metrics_df.to_csv(csv_path, index=False)
        print(f"✅ Final processed data saved to '{csv_path}'")
        
        # Generate the plots
        generate_plots(final_metrics_df, output_dir)
