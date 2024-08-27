# Use the PyTorch image as the base image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the required Python packages
RUN apt-get update && apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/* &&\
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-build-isolation auto-gptq && \
    ln -s /usr/bin/python3 /usr/bin/python

# Copy the inference script, utils.py, and dataset folder into the container
COPY inference.py utils.py data/ ./

# Set the default command to run the inference script
CMD ["python", "inference.py"]