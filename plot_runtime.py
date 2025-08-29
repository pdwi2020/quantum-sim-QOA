import pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = "final_paper_data.csv"
OUTPUT_PNG = "runtime_vs_qubits_plot.png"

def main():
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: Run 'python3 collect_final_results.py' first.")
        return

    runtime_df = df[['qubits', 'qaoa_runtime_sec']].dropna()
    agg_results = runtime_df.groupby('qubits')['qaoa_runtime_sec'].agg(['mean', 'std']).reset_index()
    agg_results['std'] = agg_results['std'].fillna(0)

    print("\nAggregated Runtime Data:")
    print(agg_results)

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.bar(
        agg_results['qubits'].astype(str),
        agg_results['mean'],
        yerr=agg_results['std'], capsize=5, color='royalblue', width=0.6
    )

    ax.set_title('Vertex AI Job Runtime vs. System Size on A100 GPU', fontsize=16, fontweight='bold')
    ax.set_xlabel('Number of Qubits', fontsize=14)
    ax.set_ylabel('Total Job Execution Time (seconds)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylim(0, agg_results['mean'].max() * 1.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    print(f"\n✅ Plot saved as '{OUTPUT_PNG}'")
    plt.show()

if __name__ == "__main__":
    main()
