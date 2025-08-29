from google.cloud import aiplatform

# ==============================================================================
# GCP AND JOB CONFIGURATION
# ==============================================================================
PROJECT_ID = "my-quantum-project-469717"
REGION = "us-central1"
BUCKET_URI = "gs://my-quantum-project-469717-bucket"
JOB_NAME = "optimized-quantum-sim-p2-run" # New job name for the deeper circuit
IMAGE_URI = "us-central1-docker.pkg.dev/my-quantum-project-469717/my-repo/quantum-sim:v5-optimized"

# --- Machine Specs (Unchanged) ---
REPLICA_COUNT = 1 # A single powerful node is sufficient for this optimization
MACHINE_TYPE = "a2-highgpu-1g"
ACCELERATOR_TYPE = "NVIDIA_TESLA_A100"
ACCELERATOR_COUNT = 1

def main():
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

    # Define the worker pool spec
    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": MACHINE_TYPE,
                "accelerator_type": ACCELERATOR_TYPE,
                "accelerator_count": ACCELERATOR_COUNT,
            },
            "replica_count": REPLICA_COUNT,
            "container_spec": {
                "image_uri": IMAGE_URI,
                # *** KEY CHANGE IS HERE ***
                # Pass environment variables to the container
                "env": [
                    {
                        "name": "NUM_CITIES",
                        "value": "4", # Let's stick with 4 cities for this intensive optimization
                    },
                    {
                        "name": "P_LAYERS",
                        "value": "2", # Run with a depth of p=2
                    },
                ],
            },
        }
    ]

    # Create and run the Custom Job
    job = aiplatform.CustomJob(
        display_name=JOB_NAME,
        worker_pool_specs=worker_pool_specs,
    )

    print(f"Submitting optimization job for 4 cities with p=2...")
    job.run()
    print("Job submitted. Check the Google Cloud Console for progress.")

if __name__ == "__main__":
    main()
