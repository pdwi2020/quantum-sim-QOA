import pandas as pd
import re
import time
from google.cloud import aiplatform, logging # <-- IMPORT ADDED HERE
from google.cloud.aiplatform.compat.types import job_state_v1 as job_state

PROJECT_ID = "my-quantum-project-469717"
REGION = "us-central1"
JOB_NAME_PREFIXES = ["paper-exp-"]

def get_job_metrics(job: aiplatform.CustomJob) -> dict:
    """Queries Cloud Logging to find metrics for a job."""
    print(f"  Querying logs for job: {job.display_name}...")
    logging_client = logging.Client(project=PROJECT_ID)
    job_id = job.name.split('/')[-1]
    log_filter = f'resource.type="ml_job" AND resource.labels.job_id="{job_id}"'
    
    metrics = { "approximation_ratio": None, "p_valid": None, "best_prob": None, "classical_runtime_sec": None }
    
    try:
        time.sleep(5) 
        entries = logging_client.list_entries(filter_=log_filter, page_size=2000)
        full_log = "\n".join([entry.payload for entry in entries if isinstance(entry.payload, str)])

        ratio_match = re.search(r"Approximation Ratio: (\d+\.\d+)", full_log)
        if ratio_match: metrics["approximation_ratio"] = float(ratio_match.group(1))

        p_valid_match = re.search(r"Feasibility \(p=\d+\): p_valid=([\d.e-]+),", full_log)
        if p_valid_match: metrics["p_valid"] = float(p_valid_match.group(1))
        
        c_runtime_match = re.search(r"Classical solver finished in ([\d.]+) seconds.", full_log)
        if c_runtime_match: metrics["classical_runtime_sec"] = float(c_runtime_match.group(1))

        if ratio_match: print(f"  --> Found metrics for {job.display_name}")
        else: print(f"  --> No approximation ratio found for {job.display_name}, likely an invalid tour result.")
    except Exception as e:
        print(f"  Could not retrieve logs due to an error: {e}")
    return metrics

def main():
    aiplatform.init(project=PROJECT_ID, location=REGION)
    jobs = aiplatform.CustomJob.list(order_by="create_time desc")
    results = []
    
    for job in jobs:
        if any(job.display_name.startswith(prefix) for prefix in JOB_NAME_PREFIXES) and job.state == job_state.JobState.JOB_STATE_SUCCEEDED:
            runtime_seconds = (job.end_time - job.start_time).total_seconds()
            
            try:
                parts = job.display_name.split('-')
                num_cities = int(parts[2])
                qubits = num_cities**2
                seed = int(parts[-1])
            except (ValueError, IndexError):
                num_cities, qubits, seed = None, None, None

            metrics = get_job_metrics(job)
            
            results.append({
                "job_name": job.display_name,
                "cities": num_cities,
                "qubits": qubits,
                "seed": seed,
                "qaoa_runtime_sec": runtime_seconds,
                **metrics
            })

    if not results:
        print("\nNo successful jobs found to collect.")
        return

    df = pd.DataFrame(results)
    output_filename = "final_paper_data.csv"
    df.to_csv(output_filename, index=False)
    print(f"\n✅ Final, corrected data saved to {output_filename}")
    print("\n--- Data Summary ---")
    print(df)


if __name__ == "__main__":
    main()
