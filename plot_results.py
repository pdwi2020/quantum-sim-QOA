import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Load and Process the Data ---
try:
    df = pd.read_csv('benchmark_results.csv')
    print("Benchmark results loaded successfully:")
    print(df)
except FileNotFoundError:
    print("Error: benchmark_results.csv not found. Please ensure the file is in the same directory.")
    exit()

# It's good practice to average results for the same problem size
# Group by 'qubits' and calculate the mean runtime
results_agg = df.groupby('qubits')['runtime_seconds'].mean().reset_index()
print("\nAggregated results (averaging runs with same qubit count):")
print(results_agg)

# --- 2. Create the Plot ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(10, 7))

# Create the bar chart
bars = ax.bar(
    results_agg['qubits'].astype(str), # Use qubits as string for categorical x-axis
    results_agg['runtime_seconds'],
    color='forestgreen',
    width=0.5,
    label='Total Job Runtime' # Updated label for clarity
)

# --- 3. Add Professional Formatting ---
ax.set_title('Vertex AI Job Runtime vs. System Size on A100 GPU', fontsize=16, fontweight='bold')
ax.set_xlabel('Number of Qubits', fontsize=14)
ax.set_ylabel('Total Job Execution Time (seconds)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)

# Add data labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}s', ha='center', va='bottom', fontsize=12)

# Set y-axis limit to give some space above the tallest bar
ax.set_ylim(0, results_agg['runtime_seconds'].max() * 1.2)

# --- 4. Save and Show the Plot ---
plt.tight_layout()
plt.savefig('runtime_vs_qubits.png', dpi=300)
plt.show()

print("\n✅ Plot saved as 'runtime_vs_qubits.png'")
