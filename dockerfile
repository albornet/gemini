# Use the PyTorch image as the base image
FROM pytorch/pytorch:2.4.0-cuda12.2-cudnn9-runtime

# Set the working directory in the container
WORKDIR /app

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt