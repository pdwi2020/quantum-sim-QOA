from google.cloud import aiplatform

# ==============================================================================
# GCP AND JOB CONFIGURATION
# ==============================================================================
PROJECT_ID = "my-quantum-project-469717"
REGION = "us-central1"
BUCKET_URI = "gs://my-quantum-project-469717-bucket" 
IMAGE_URI = "us-central1-docker.pkg.dev/my-quantum-project-469717/my-repo/quantum-sim:v4-distributed"

# --- Machine Specs ---
REPLICA_COUNT = 1 
MACHINE_TYPE = "a2-highgpu-1g"
ACCELERATOR_TYPE = "NVIDIA_TESLA_A100"
ACCELERATOR_COUNT = 1

# --- EXPERIMENT PARAMETERS ---
CITIES_TO_TEST = [4, 5] 

def main():
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

    for num_cities in CITIES_TO_TEST:
        job_name = f"quantum-sim-{num_cities}-cities-run"
        print(f"--- Submitting job for {num_cities} cities: {job_name} ---")

        container_spec = {
            "image_uri": IMAGE_URI,
            "command": [],
            "args": [],
            "env": [
                {
                    "name": "NUM_CITIES",
                    "value": str(num_cities)
                }
            ]
        }

        worker_pool_specs = [
            {
                "machine_spec": {
                    "machine_type": MACHINE_TYPE,
                    "accelerator_type": ACCELERATOR_TYPE,
                    "accelerator_count": ACCELERATOR_COUNT,
                },
                "replica_count": REPLICA_COUNT,
                "container_spec": container_spec,
            }
        ]

        job = aiplatform.CustomJob(
            display_name=job_name,
            worker_pool_specs=worker_pool_specs,
        )
        
        job.submit()
        
        # *** CORRECTION IS HERE ***
        # Manually construct the web UI link
        job_id = job.name
        web_ui_link = f"https://console.cloud.google.com/ai/platform/locations/{REGION}/training/{job_id}?project={PROJECT_ID}"
        
        print(f"✅ Job {job_name} submitted. View at: {web_ui_link}")
        print("-" * 50)

if __name__ == "__main__":
    main()
