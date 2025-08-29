import os
import subprocess
from google.cloud import aiplatform
from google.cloud.aiplatform.compat.types import job_state_v1 as job_state

# ==============================================================================
# CONFIGURATION
# ==============================================================================
PROJECT_ID = "my-quantum-project-469717"
REGION = "us-central1"
JOB_NAME = "final-analysis-run"

def main():
    """Finds the latest successful job and downloads its artifacts."""
    
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    print(f"Searching for latest successful job named '{JOB_NAME}'...")
    
    job_filter = f'display_name="{JOB_NAME}"'
    jobs = aiplatform.CustomJob.list(filter=job_filter, order_by="create_time desc")
    
    latest_successful_job = None
    for job in jobs:
        if job.state == job_state.JobState.JOB_STATE_SUCCEEDED:
            latest_successful_job = job
            break

    if not latest_successful_job:
        print(f"Error: No successful job named '{JOB_NAME}' found.")
        return

    print(f"Found job: {latest_successful_job.name}")
    
    # *** CORRECTION IS HERE ***
    # Access the output directory through the job's underlying gca_resource
    try:
        output_directory = latest_successful_job._gca_resource.job_spec.base_output_directory.output_uri_prefix
    except AttributeError:
        print("Error: Could not find the GCS output directory in the job's specification.")
        return
        
    source_path = os.path.join(output_directory, "model", "outputs")
    destination_path = "./final_project_artifacts"
    
    print(f"\nDownloading artifacts from: {source_path}")
    print(f"Saving to local directory: {destination_path}")

    command = ["gsutil", "-m", "cp", "-r", source_path, destination_path]
    
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"\n✅ Success! All graphs and CSVs downloaded to '{destination_path}' directory.")
    except subprocess.CalledProcessError as e:
        print("\n--- DOWNLOAD FAILED ---")
        print(f"Stderr: {e.stderr}")
        print("This can happen if the 'outputs' directory was not created by the job.")
    except FileNotFoundError:
        print("\nError: 'gsutil' command not found. Please ensure the Google Cloud SDK is installed and in your PATH.")

if __name__ == "__main__":
    main()
