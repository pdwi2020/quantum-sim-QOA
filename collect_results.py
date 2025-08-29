import pandas as pd
from google.cloud import aiplatform
from google.cloud.aiplatform.compat.types import job_state_v1 as job_state

# ==============================================================================
# CONFIGURATION
# ==============================================================================
PROJECT_ID = "my-quantum-project-469717"
REGION = "us-central1"
# This prefix is used to find all your relevant experiment jobs
JOB_NAME_PREFIX = "quantum-sim-"

def main():
    """Finds all successful jobs and extracts their benchmark data."""
    
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    print(f"Fetching completed jobs with prefix '{JOB_NAME_PREFIX}'...")
    
    # List all CustomJob objects in the project
    jobs = aiplatform.CustomJob.list()
    
    results = []
    
    for job in jobs:
        # Check if the job name matches our experiment and if it succeeded
        if job.display_name.startswith(JOB_NAME_PREFIX) and job.state == job_state.JobState.JOB_STATE_SUCCEEDED:
            
            # Extract number of cities from the display name
            try:
                # e.g., "quantum-sim-4-cities-run" -> "4"
                parts = job.display_name.split('-')
                num_cities = int(parts[2])
                n_qubits = num_cities**2
                
                # Calculate runtime
                runtime = job.end_time - job.start_time
                runtime_seconds = runtime.total_seconds()
                
                results.append({
                    "cities": num_cities,
                    "qubits": n_qubits,
                    "runtime_seconds": runtime_seconds,
                    "job_id": job.name
                })
                print(f"Found successful run for {num_cities} cities. Runtime: {runtime_seconds:.2f}s")
                
            except (ValueError, IndexError):
                print(f"Could not parse job name: {job.display_name}")

    if not results:
        print("\nNo successful jobs found to collect.")
        return

    # Create a pandas DataFrame for easy viewing and saving
    df = pd.DataFrame(results)
    df = df.sort_values(by="qubits")

    print("\n--- Benchmark Results ---")
    print(df.to_string(index=False))
    
    # Save results to a CSV file
    output_filename = "benchmark_results.csv"
    df.to_csv(output_filename, index=False)
    print(f"\n✅ Results saved to {output_filename}")

if __name__ == "__main__":
    main()
