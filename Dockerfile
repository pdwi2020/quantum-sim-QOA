FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-12
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src
# Point to the distributed script
ENTRYPOINT ["python", "src/distributed_main.py"]
