# Use the PyTorch image as the base image
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Install the required packages, CUDA dependencies, and Python packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python

# Copy requirements to app folder
COPY requirements.txt /app/

# Install the Python packages listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install auto-gptq separately since it only works properly if built from source
RUN TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0" pip install git+https://github.com/AutoGPTQ/AutoGPTQ.git

# Set an environment variable for huggingface login
ENV HF_TOKEN=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

# Set the default command to run the inference script
CMD ["python", "inference.py"]
# docker run -v .:/app -v ~/.cache/huggingface -it --gpus all gemini-image:latest /bin/bash