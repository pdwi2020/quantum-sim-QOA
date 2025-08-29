from kfp.dsl import component, Input, Output, Dataset

# This is the full image name of the container.
# We'll use v2 after we rebuild the container with the new main.py
PIPELINE_IMAGE = "us-central1-docker.pkg.dev/my-quantum-project-469717/my-repo/quantum-sim:v2"

@component(
    base_image=PIPELINE_IMAGE
)
def generate_problem_task(
    num_cities: int,
    pauli_terms_dataset: Output[Dataset]
):
    # This component is now just a reference to a command
    # inside the base_image. KFP will run this for us.
    # The actual implementation is in main.py's "generate" task.
    pass

@component(
    base_image=PIPELINE_IMAGE
)
def run_simulation_task(
    pauli_terms_dataset: Input[Dataset],
    p_layers: int
):
    # This component is also a reference to a command.
    # The implementation is in main.py's "simulate" task.
    pass
