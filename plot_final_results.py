import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Load and Process the Data ---
try:
    df = pd.read_csv('final_results.csv')
    print("Final results loaded successfully:")
    print(df)
except FileNotFoundError:
    print("Error: final_results.csv not found.")
    print("Please run collect_final_results.py and manually add the approximation ratio first.")
    exit()

# --- 2. Create Plot 1: Runtime vs. Qubits ---
runtime_data = df.dropna(subset=['runtime_seconds'])
runtime_agg = runtime_data.groupby('qubits')['runtime_seconds'].mean().reset_index()

plt.style.use('seaborn-v0_8-darkgrid')
fig1, ax1 = plt.subplots(figsize=(10, 7))

bars1 = ax1.bar(
    runtime_agg['qubits'].astype(str),
    runtime_agg['runtime_seconds'],
    color='royalblue',
    width=0.5
)
ax1.set_title('Vertex AI Job Runtime vs. System Size on A100 GPU', fontsize=16, fontweight='bold')
ax1.set_xlabel('Number of Qubits', fontsize=14)
ax1.set_ylabel('Total Job Execution Time (seconds)', fontsize=14)
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}s', ha='center', va='bottom')
ax1.set_ylim(0, runtime_agg['runtime_seconds'].max() * 1.2)
plt.tight_layout()
plt.savefig('final_runtime_vs_qubits.png', dpi=300)
print("\n✅ Runtime plot saved as 'final_runtime_vs_qubits.png'")

# --- 3. Create Plot 2: Solution Quality (Approximation Ratio) ---
optimizer_data = df.dropna(subset=['approximation_ratio'])

if not optimizer_data.empty:
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    ratio = optimizer_data.iloc[0]['approximation_ratio']
    qubits = int(optimizer_data.iloc[0]['qubits'])

    ax2.bar(
        [f'{qubits} Qubits'],
        [ratio],
        color='forestgreen',
        width=0.4,
        label='QAOA Result'
    )
    # Add a line for the ideal result
    ax2.axhline(y=1.0, color='r', linestyle='--', label='Optimal Solution (Ratio = 1.0)')
    
    ax2.set_title(f'QAOA Approximation Ratio for {qubits}-Qubit TSP', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Approximation Ratio (Cost_QAOA / Cost_Optimal)', fontsize=12)
    ax2.set_ylim(0, max(1.2, ratio * 1.2))
    ax2.legend()
    
    # Add data label
    ax2.text(0, ratio, f'{ratio:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('approximation_ratio.png', dpi=300)
    print("✅ Approximation ratio plot saved as 'approximation_ratio.png'")
else:
    print("\nNo approximation ratio data found to plot. Please complete final_results.csv.")

plt.show()
