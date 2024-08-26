# Use the PyTorch image as the base image
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt