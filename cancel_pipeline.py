from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(project="998503420879", location="us-central1")

# Get the running pipeline job
pipeline_job = aiplatform.PipelineJob.get(
    resource_name="projects/998503420879/locations/us-central1/pipelineJobs/quantum-simulation-pipeline-20250822003655"
)

# Cancel the job
pipeline_job.cancel()
print("✅ Pipeline job cancellation requested.")
