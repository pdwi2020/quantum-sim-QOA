from google.cloud import aiplatform

PROJECT_ID = "my-quantum-project-469717"
REGION = "us-central1"
BUCKET_URI = "gs://my-quantum-project-469717-bucket" 
IMAGE_URI = "us-central1-docker.pkg.dev/my-quantum-project-469717/my-repo/quantum-sim:v14-definitive"

MACHINE_TYPE = "a2-highgpu-1g"
ACCELERATOR_TYPE = "NVIDIA_TESLA_A100"
ACCELERATOR_COUNT = 1

# Define the final experiments for the paper
EXPERIMENTS = [
    {
        "name": "4-cities-p1-5-cobyla",
        "env": { "NUM_CITIES": "4", "P_VALUES": "1,2,3,4,5", "OPTIMIZER": "COBYLA" }
    },
    {
        "name": "4-cities-p1-5-diff-evol",
        "env": { "NUM_CITIES": "4", "P_VALUES": "1,2,3,4,5", "OPTIMIZER": "DIFFERENTIAL_EVOLUTION" }
    }
]

def main():
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)
    for exp in EXPERIMENTS:
        job_name = f"paper-exp-{exp['name']}"
        print(f"\n--- Submitting job: {job_name} ---")
        env_vars = [{"name": k, "value": v} for k, v in exp['env'].items()]
        worker_pool_specs = [{
            "machine_spec": {"machine_type": MACHINE_TYPE, "accelerator_type": ACCELERATOR_TYPE, "accelerator_count": ACCELERATOR_COUNT},
            "replica_count": 1,
            "container_spec": {"image_uri": IMAGE_URI, "env": env_vars},
        }]
        job = aiplatform.CustomJob(display_name=job_name, worker_pool_specs=worker_pool_specs)
        
        # *** KEY CHANGE IS HERE ***
        # .run() waits for the job to complete before continuing the loop
        job.run()
        
        print(f"✅ Job {job_name} finished.")

    print("\nAll sequential jobs complete.")

if __name__ == "__main__":
    main()
