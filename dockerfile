# Use the PyTorch image as the base image
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Set the CUDA architecture list for auto-gptq installation
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6"

# Set the working directory in the container
WORKDIR /app

# Install the required packages, CUDA dependencies, and Python packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python

# Copy the source code and other necessary files into the container
COPY requirements.txt inference.py utils.py data/ ./

# Install the Python packages listed in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir git+https://github.com/AutoGPTQ/AutoGPTQ.git

# Set the default command to run the inference script
CMD ["python", "inference.py"]