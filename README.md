# quantum-sim-QOA

# A Cloud-Native Platform for Scalable Quantum Simulation and Landscape Analysis

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![Frameworks](https://img.shields.io/badge/Frameworks-PyTorch%20%7C%20GCP%20Vertex%20AI-orange.svg)
![Status](https://img.shields.io/badge/status-complete-success.svg)

This repository contains the complete source code and research artifacts for the project "A Cloud-Native Platform for Probing the Deceptive Energy Landscapes of Quantum Optimization Algorithms." We present the design, implementation, and deployment of a scalable, cloud-native platform on Google Cloud Platform (GCP) for high-performance quantum simulation.

The platform was used to investigate the Quantum Approximate Optimization Algorithm (QAOA) for the Traveling Salesman Problem (TSP). Our key scientific finding is that the QAOA energy landscape for this problem is fundamentally "deceptive," consistently trapping standard classical optimizers in low-energy but physically invalid solutions.

---

## 🔑 Key Features

*   **Truly Matrix-Free Simulator:** An advanced quantum simulator implemented in PyTorch that overcomes traditional memory walls by using analytical Pauli evolution, scaling with the state vector ($O(2^N)$) instead of the operator matrix ($O(4^N)$).
*   **Scalable Cloud-Native Architecture:** A fully automated, serverless MLOps workflow built on Google Cloud using Docker, Cloud Build, Artifact Registry, and Vertex AI.
*   **High-Performance Execution:** Capable of simulating 16-qubit and 25-qubit systems with an end-to-end job runtime of ~60 seconds on a single NVIDIA A100 GPU, demonstrating that the workflow is infrastructure-limited, not compute-limited.
*   **Automated Scientific Benchmarking:** The entire platform is designed as a tool for reproducible research, enabling automated, multi-job experiments to probe complex algorithmic behaviors.
*   **Discovery of Deceptive Landscapes:** Provides a concrete, real-world example and platform for studying a key challenge in variational quantum algorithms, where the optimization landscape misleads classical optimizers.

---

## 🏗️ System Architecture and MLOps Workflow

Our platform is designed for automation and scalability, leveraging a serverless MLOps architecture on GCP. The overall system architecture (Figure 1) proceeds from a local development blueprint to a fully managed cloud execution environment. This is orchestrated by a formal MLOps pipeline (Figure 2) that ensures reproducibility and automates the entire experimental lifecycle.

### Figure 1: Overall System Architecture
![Overall System Architecture](images/Figure%201%20Overall%20System%20Architecture.png)

### Figure 2: MLOps Workflow Pipeline
![MLOps Workflow Pipeline](images/Figure%202%20MLOps%20Workflow%20Pipeline.png)

---

## 🧠 The Scientific Challenge: A Hierarchy of Memory Walls

The primary obstacle to classical quantum simulation is the exponential scaling of resources. Our investigation revealed a hierarchy of cascading bottlenecks that must be overcome to enable large-scale simulation.

### Figure 3: Memory Bottleneck Hierarchy and Resolution
![Memory Bottleneck Hierarchy and Resolution](images/Figure%203%20Memory%20Bottleneck%20Hierarchy%20and%20Resolution.png)

1.  **Level 1: GPU VRAM Limitation:** A naive approach fails at 16 qubits, as the dense operator matrix (~64GB) and its temporary copies exceed the VRAM of high-end GPUs.
2.  **Level 2: CPU System RAM Limitation:** JIT methods that stream to the GPU fail next, as classical libraries exhaust host system RAM when constructing the matrices.
3.  **Level 3: Matrix-Free Resolution:** Our solution avoids constructing large matrices, reducing memory complexity from O(4^N) to O(2^N).
4.  **Level 4: The Algorithmic Challenge:** With memory issues solved, the platform reveals the final, most profound challenge: the difficulty of the classical optimization task itself.

---

## ⚛️ The Solution: A Truly Matrix-Free Simulator

The core innovation of our platform is the `PauliEvolutionSimulator`, which avoids all dense matrix construction. The Hamiltonian is stored as a list of Pauli strings and coefficients, and the evolution is computed analytically on the GPU using memory-efficient bitwise permutations.

### Figure 4: Matrix-Free Quantum Simulator Architecture
![Matrix-Free Quantum Simulator Architecture](images/Figure%204%20Matrix-Free%20Quantum%20Simulator%20Architecture.png)

---

## 🔬 The Experiment: QAOA for the Traveling Salesman Problem

We used our platform to conduct a comprehensive optimization campaign for the 4-city (16-qubit) TSP.

### Figure 6: TSP Hamiltonian Formulation
![TSP Hamiltonian Formulation](images/Figure%206%20TSP%20Hamiltonian%20Formulation.png)

The experiment followed a robust, multi-start optimization workflow designed to find the best possible QAOA solution.

### Figure 5: QAOA Optimization Workflow
![QAOA Optimization Workflow](images/Figure%205%20QAOA%20Optimization%20Workflow.png)

---

## 📊 Results and Key Findings

### Platform Performance

Our platform demonstrates extreme efficiency. The total job runtime is dominated by the fixed ~60-second overhead of cloud infrastructure, with the actual compute time for the quantum simulation being negligible in comparison for systems up to 25 qubits.

![Vertex AI Job Runtime vs. System Size on A100 GPU](images/Vertex%20AI%20Job%20Runtime%20vs.%20System%20Size%20on%20A100%20GPU.png)

### The Deceptive Landscape Discovery

Our primary scientific finding is that for the 4-city TSP, the QAOA energy landscape is "deceptive." Despite a robust experimental setup, the classical optimizer consistently converged to low-energy states that corresponded to physically **invalid** solutions.

A deceptive landscape is a rugged, multi-modal energy surface where the lowest points (minima) do not correspond to the true, valid solution of the problem. As shown conceptually in Figure 8, this can easily trap a standard optimizer.

**Figure 8: Conceptual Visualization of a Deceptive Landscape**
![Conceptual Visualization of a Deceptive Landscape](images/deceptive_landscape_3d.png)
> The optimizer's goal is to find the lowest point (global minimum), but the rugged terrain with many local minima can trap it in a suboptimal solution.

Our empirical results, visualized in Figure 9, confirmed this behavior. The optimizer successfully finds low-energy states, but the landscape structure, particularly for deeper circuits, is complex and contains many such traps. Ultimately, the states with the highest probability were all invalid.

**Figure 9: Empirical Evidence of a Deceptive Landscape**
![Empirical Evidence of a Deceptive Landscape](images/QAOA%20Deceptive%20Optimization%20Landscape.png)
> The plot shows the energy landscape for different QAOA depths (p). The optimizer aims for the global minimum (star), but local minima (X) can trap the search, especially as the landscape becomes more rugged with increasing depth.

---

## 🚀 Getting Started: How to Run This Project

Follow these steps to replicate our experiments on Google Cloud Platform.

### Prerequisites
1.  **Google Cloud SDK:** Install and initialize (`gcloud init`).
2.  **Docker:** Install Docker Desktop for your OS.
3.  **GCP Project:** Have a GCP project with billing enabled. Enable the **Vertex AI API**, **Cloud Build API**, and **Artifact Registry API**.
4.  **Authentication:**
    ```bash
    gcloud auth login
    gcloud auth application-default login
    gcloud config set project YOUR_GCP_PROJECT_ID
    ```

### Step-by-Step Guide

1.  **Clone the Repository**
    ```bash
    git clone <your-repo-url>
    cd quantum-sim
    ```

2.  **Configure Your Project**
    *   **Create a GCS Bucket:**
        ```bash
        gsutil mb -p YOUR_GCP_PROJECT_ID -l us-central1 gs://your-unique-bucket-name
        ```
    *   **Edit `run_optimizer.py`:**
        *   Replace `my-quantum-project-469717` with your `YOUR_GCP_PROJECT_ID`.
        *   Replace `gs://my-quantum-project-469717-bucket` with your `gs://your-unique-bucket-name`.

3.  **Set Up Artifact Registry**
    ```bash
    gcloud artifacts repositories create my-repo \
      --repository-format=docker \
      --location=us-central1 \
      --description="Quantum Simulation Repo"
    ```

4.  **Build and Push the Container using Cloud Build**
    This command sends your code to Cloud Build, which builds the container remotely and pushes it to Artifact Registry.
    ```bash
    gcloud builds submit --tag us-central1-docker.pkg.dev/YOUR_GCP_PROJECT_ID/my-repo/quantum-sim:v11-definitive .
    ```
    *Note: Make sure the tag in this command matches the `IMAGE_URI` in your `run_optimizer.py` script.*

5.  **Launch the Definitive Vertex AI Job**
    This script will launch the full experiment on Vertex AI, which runs `src/optimizer_main.py` inside the container to generate all data and plots.
    ```bash
    python3 run_optimizer.py
    ```

6.  **Monitor and Collect Results**
    *   Monitor the job's progress in the **Google Cloud Console** under **Vertex AI > Training > Custom Jobs**.
    *   After the job completes successfully, the output artifacts (plots and CSV files) will be available in your GCS bucket.

---
