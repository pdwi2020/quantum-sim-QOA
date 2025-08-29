from google.cloud import aiplatform

PROJECT_ID = "my-quantum-project-469717"
REGION = "us-central1"
BUCKET_URI = "gs://my-quantum-project-469717-bucket" 
JOB_NAME = "final-analysis-run"
IMAGE_URI = "us-central1-docker.pkg.dev/my-quantum-project-469717/my-repo/quantum-sim:v12-final-analysis"

MACHINE_TYPE = "a2-highgpu-1g"
ACCELERATOR_TYPE = "NVIDIA_TESLA_A100"
ACCELERATOR_COUNT = 1

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
                "env": [
                    { "name": "NUM_CITIES", "value": "4" },
                    { "name": "P_VALUES", "value": "1,2,3" },
                    { "name": "NUM_STARTS", "value": "5" },
                    { "name": "MAXITER", "value": "200" },
                ],
            },
        }
    ]

    job = aiplatform.CustomJob(
        display_name=JOB_NAME,
        worker_pool_specs=worker_pool_specs,
    )

    print(f"Submitting final analysis job...")
    job.run()

if __name__ == "__main__":
    main()
