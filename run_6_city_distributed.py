from google.cloud import aiplatform

PROJECT_ID = "my-quantum-project-469717"
REGION = "us-central1"
BUCKET_URI = "gs://my-quantum-project-469717-bucket" 
JOB_NAME = "6-city-distributed-run-final-attempt" # New name
IMAGE_URI = "us-central1-docker.pkg.dev/my-quantum-project-469717/my-repo/quantum-sim:v15-distributed-final"
SERVICE_ACCOUNT = "998503420879-compute@developer.gserviceaccount.com"

# --- Machine Specs ---
MACHINE_TYPE = "a2-highgpu-4g" 
ACCELERATOR_TYPE = "NVIDIA_TESLA_A100"
ACCELERATOR_COUNT = 4

def main():
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)
    
    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": MACHINE_TYPE,
                "accelerator_type": ACCELERATOR_TYPE,
                "accelerator_count": ACCELERATOR_COUNT,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": IMAGE_URI, 
                "env": [{"name": "NUM_CITIES", "value": "6"}]
            },
        }
    ]
    job = aiplatform.CustomJob(display_name=JOB_NAME, worker_pool_specs=worker_pool_specs)
    
    print(f"Submitting single-node, 4-GPU distributed job for 6 cities with extended timeout...")
    
    # *** KEY CHANGE IS HERE: Adding a timeout and specifying the service account ***
    job.run(
        service_account=SERVICE_ACCOUNT,
        timeout=72000 # Timeout in seconds (e.g., 72000s = 20 hours)
    )

if __name__ == "__main__":
    main()
