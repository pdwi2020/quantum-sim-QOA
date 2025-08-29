#!/usr/bin/env python3
"""
Final results processing script for quantum TSP experiments.
Monitors jobs, fetches logs, generates CSVs and graphs when jobs complete.
"""

import os
import time
import json
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from pathlib import Path
import re

# Job configuration
JOBS = [
    {
        "name": "paper-exp-4-cities-p1-5-cobyla",
        "job_id": "6178438221983121408",
        "optimizer": "COBYLA",
        "config": {"NUM_CITIES": "4", "P_VALUES": "1,2,3,4,5", "OPTIMIZER": "COBYLA"}
    },
    {
        "name": "paper-exp-4-cities-p1-5-diff-evol", 
        "job_id": "4194602586126417920",
        "optimizer": "DIFFERENTIAL_EVOLUTION",
        "config": {"NUM_CITIES": "4", "P_VALUES": "1,2,3,4,5", "OPTIMIZER": "DIFFERENTIAL_EVOLUTION"}
    }
]

PROJECT_ID = "my-quantum-project-469717"
REGION = "us-central1"
RESULTS_DIR = Path("final_paper_results")

class JobProcessor:
    def __init__(self):
        self.results_dir = RESULTS_DIR
        self.setup_directories()
    
    def setup_directories(self):
        """Create organized directory structure for results."""
        dirs = ["logs", "csvs", "graphs", "raw_data", "summary"]
        for dir_name in dirs:
            (self.results_dir / dir_name).mkdir(parents=True, exist_ok=True)
        print(f"✅ Results directory structure created: {self.results_dir}")
    
    def check_job_status(self, job_id):
        """Check the status of a specific job."""
        try:
            cmd = f"gcloud ai custom-jobs describe {job_id} --region={REGION} --format=json"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                job_info = json.loads(result.stdout)
                return job_info.get('state', 'UNKNOWN')
            return 'ERROR'
        except Exception as e:
            print(f"❌ Error checking job {job_id}: {e}")
            return 'ERROR'
    
    def fetch_job_logs(self, job_id, job_name):
        """Fetch logs for a completed job."""
        try:
            print(f"📥 Fetching logs for {job_name}...")
            cmd = f"gcloud ai custom-jobs stream-logs {job_id} --region={REGION}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                log_file = self.results_dir / "logs" / f"{job_name}.log"
                with open(log_file, 'w') as f:
                    f.write(result.stdout)
                print(f"✅ Logs saved to {log_file}")
                return result.stdout
            else:
                print(f"❌ Failed to fetch logs for {job_name}")
                return None
        except Exception as e:
            print(f"❌ Error fetching logs for {job_name}: {e}")
            return None
    
    def parse_experiment_results(self, logs, job_name, optimizer):
        """Parse experiment results from logs."""
        results = []
        
        # Pattern to match experiment results
        # Looking for patterns like: "P=1: Best cost: 123.45, Iterations: 100"
        patterns = [
            r'P=(\d+).*?Best cost:\s*([\d.]+).*?Iterations:\s*(\d+)',
            r'p=(\d+).*?cost:\s*([\d.]+).*?iterations:\s*(\d+)',
            r'P_VALUE:\s*(\d+).*?FINAL_COST:\s*([\d.]+).*?ITERATIONS:\s*(\d+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, logs, re.IGNORECASE | re.DOTALL)
            for match in matches:
                p_value, cost, iterations = match
                results.append({
                    'job_name': job_name,
                    'optimizer': optimizer,
                    'p_value': int(p_value),
                    'best_cost': float(cost),
                    'iterations': int(iterations),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Also look for convergence data if available
        convergence_pattern = r'Iteration (\d+): Cost = ([\d.]+)'
        convergence_matches = re.findall(convergence_pattern, logs, re.IGNORECASE)
        
        convergence_data = []
        for match in convergence_matches:
            iteration, cost = match
            convergence_data.append({
                'iteration': int(iteration),
                'cost': float(cost)
            })
        
        return results, convergence_data
    
    def save_to_csv(self, results, convergence_data, job_name):
        """Save results to CSV files."""
        if results:
            # Main results CSV
            df_results = pd.DataFrame(results)
            csv_file = self.results_dir / "csvs" / f"{job_name}_results.csv"
            df_results.to_csv(csv_file, index=False)
            print(f"✅ Results saved to {csv_file}")
            
            # Raw data JSON for detailed analysis
            raw_file = self.results_dir / "raw_data" / f"{job_name}_raw.json"
            with open(raw_file, 'w') as f:
                json.dump({
                    'results': results,
                    'convergence': convergence_data
                }, f, indent=2)
            print(f"✅ Raw data saved to {raw_file}")
            
        if convergence_data:
            # Convergence CSV
            df_conv = pd.DataFrame(convergence_data)
            df_conv['job_name'] = job_name
            conv_file = self.results_dir / "csvs" / f"{job_name}_convergence.csv"
            df_conv.to_csv(conv_file, index=False)
            print(f"✅ Convergence data saved to {conv_file}")
    
    def generate_graphs(self):
        """Generate comprehensive graphs from all collected data."""
        print("📊 Generating graphs...")
        
        # Collect all results
        all_results = []
        all_convergence = []
        
        for csv_file in (self.results_dir / "csvs").glob("*_results.csv"):
            df = pd.read_csv(csv_file)
            all_results.append(df)
        
        for csv_file in (self.results_dir / "csvs").glob("*_convergence.csv"):
            df = pd.read_csv(csv_file)
            all_convergence.append(df)
        
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            self.create_comparison_graphs(combined_results)
        
        if all_convergence:
            combined_convergence = pd.concat(all_convergence, ignore_index=True)
            self.create_convergence_graphs(combined_convergence)
    
    def create_comparison_graphs(self, df):
        """Create comparison graphs between optimizers."""
        plt.style.use('seaborn-v0_8')
        
        # 1. Cost comparison by p-value
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        sns.barplot(data=df, x='p_value', y='best_cost', hue='optimizer', ax=ax1)
        ax1.set_title('Best Cost by P-Value and Optimizer')
        ax1.set_xlabel('P-Value')
        ax1.set_ylabel('Best Cost')
        
        # Box plot
        sns.boxplot(data=df, x='optimizer', y='best_cost', ax=ax2)
        ax2.set_title('Cost Distribution by Optimizer')
        ax2.set_xlabel('Optimizer')
        ax2.set_ylabel('Best Cost')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "graphs" / "cost_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Iterations comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df, x='p_value', y='iterations', hue='optimizer', ax=ax)
        ax.set_title('Iterations Required by P-Value and Optimizer')
        ax.set_xlabel('P-Value')
        ax.set_ylabel('Iterations')
        plt.tight_layout()
        plt.savefig(self.results_dir / "graphs" / "iterations_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Performance heatmap
        if len(df) > 1:
            pivot_cost = df.pivot(index='p_value', columns='optimizer', values='best_cost')
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(pivot_cost, annot=True, fmt='.2f', cmap='viridis', ax=ax)
            ax.set_title('Cost Heatmap: P-Value vs Optimizer')
            plt.tight_layout()
            plt.savefig(self.results_dir / "graphs" / "performance_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✅ Comparison graphs saved")
    
    def create_convergence_graphs(self, df):
        """Create convergence analysis graphs."""
        if df.empty:
            return
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for job_name in df['job_name'].unique():
            job_data = df[df['job_name'] == job_name]
            ax.plot(job_data['iteration'], job_data['cost'], 
                   label=job_name, marker='o', markersize=3)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost')
        ax.set_title('Convergence Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "graphs" / "convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Convergence graphs saved")
    
    def create_summary_report(self):
        """Create a comprehensive summary report."""
        summary_file = self.results_dir / "summary" / "experiment_summary.md"
        
        # Collect all CSV data
        all_data = []
        for csv_file in (self.results_dir / "csvs").glob("*_results.csv"):
            df = pd.read_csv(csv_file)
            all_data.append(df)
        
        if not all_data:
            print("⚠️ No data found for summary report")
            return
            
        combined_df = pd.concat(all_data, ignore_index=True)
        
        with open(summary_file, 'w') as f:
            f.write("# Quantum TSP Final Experiment Results\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Experiment Configuration\n")
            f.write("- Problem: 4-city TSP\n")
            f.write("- P-values tested: 1, 2, 3, 4, 5\n")
            f.write("- Optimizers: COBYLA, Differential Evolution\n\n")
            
            f.write("## Results Summary\n\n")
            
            # Best results per optimizer
            for optimizer in combined_df['optimizer'].unique():
                opt_data = combined_df[combined_df['optimizer'] == optimizer]
                best_result = opt_data.loc[opt_data['best_cost'].idxmin()]
                f.write(f"### {optimizer}\n")
                f.write(f"- Best cost: {best_result['best_cost']:.4f}\n")
                f.write(f"- Best P-value: {best_result['p_value']}\n")
                f.write(f"- Iterations: {best_result['iterations']}\n\n")
            
            # Statistical summary
            f.write("## Statistical Summary\n\n")
            stats = combined_df.groupby('optimizer')['best_cost'].agg(['mean', 'std', 'min', 'max'])
            f.write(stats.to_string())
            f.write("\n\n")
            
            f.write("## Files Generated\n")
            f.write("- Raw logs: `logs/` directory\n")
            f.write("- CSV data: `csvs/` directory\n")
            f.write("- Graphs: `graphs/` directory\n")
            f.write("- Raw JSON data: `raw_data/` directory\n")
        
        print(f"✅ Summary report created: {summary_file}")
    
    def monitor_and_process(self):
        """Main monitoring loop."""
        print("🚀 Starting job monitoring and processing...")
        
        completed_jobs = set()
        
        while len(completed_jobs) < len(JOBS):
            for job in JOBS:
                if job["job_id"] in completed_jobs:
                    continue
                
                status = self.check_job_status(job["job_id"])
                print(f"📊 Job {job['name']}: {status}")
                
                if status == "JOB_STATE_SUCCEEDED":
                    print(f"🎉 Job {job['name']} completed successfully!")
                    
                    # Fetch logs
                    logs = self.fetch_job_logs(job["job_id"], job["name"])
                    
                    if logs:
                        # Parse results
                        results, convergence = self.parse_experiment_results(
                            logs, job["name"], job["optimizer"]
                        )
                        
                        # Save to CSV
                        self.save_to_csv(results, convergence, job["name"])
                    
                    completed_jobs.add(job["job_id"])
                
                elif status == "JOB_STATE_FAILED":
                    print(f"❌ Job {job['name']} failed!")
                    # Still fetch logs for debugging
                    self.fetch_job_logs(job["job_id"], job["name"])
                    completed_jobs.add(job["job_id"])
            
            if len(completed_jobs) < len(JOBS):
                print(f"⏳ Waiting for {len(JOBS) - len(completed_jobs)} jobs to complete...")
                time.sleep(30)  # Check every 30 seconds
        
        print("🎯 All jobs completed! Processing final results...")
        
        # Generate graphs and summary
        self.generate_graphs()
        self.create_summary_report()
        
        print(f"✅ All processing complete! Results saved in {self.results_dir}")

def main():
    processor = JobProcessor()
    processor.monitor_and_process()

if __name__ == "__main__":
    main()
