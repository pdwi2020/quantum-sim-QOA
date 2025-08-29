import pandas as pd
import matplotlib.pyplot as plt

try:
    df = pd.read_csv('metrics_by_p.csv')
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot 1: Approximation Ratio vs. p
    ax1.plot(df['p'], df['approximation_ratio'], marker='o', linestyle='-', color='forestgreen')
    ax1.axhline(y=1.0, color='r', linestyle='--', label='Optimal Solution (Ratio = 1.0)')
    ax1.set_title('QAOA Approximation Ratio vs. Circuit Depth (p)', fontsize=16, fontweight='bold')
    ax1.set_xlabel('QAOA Depth (p)', fontsize=12)
    ax1.set_ylabel('Approximation Ratio (Lower is Better)', fontsize=12)
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Feasible Probability vs. p
    ax2.plot(df['p'], df['p_valid'], marker='o', linestyle='-', color='royalblue')
    ax2.set_title('Probability of Measuring a Valid Tour vs. Circuit Depth (p)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('QAOA Depth (p)', fontsize=12)
    ax2.set_ylabel('Feasible State Probability (p_valid)', fontsize=12)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('summary_plots.png', dpi=300)
    plt.show()
    
    print("\n✅ Final summary plot saved as 'summary_plots.png'")

except FileNotFoundError:
    print("Error: 'metrics_by_p.csv' not found. Ensure the download was successful.")

