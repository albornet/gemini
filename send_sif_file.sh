#!/bin/bash

# Define variables
IMAGE_NAME="gemini"
SERVER_ADDRESS="borneta@login2.baobab.hpc.unige.ch"
SIF_FOLDER_LOCATION="/home/users/b/borneta/sif"

# Build the Docker image
sudo docker build -t $IMAGE_NAME-image:latest .
if [ $? -ne 0 ]; then
    echo "Docker build failed. Exiting."
    exit 1
fi

# Save the Docker image to a tar file
sudo docker save $IMAGE_NAME-image:latest > $IMAGE_NAME-image.tar
if [ $? -ne 0 ]; then
    echo "Docker save failed. Exiting."
    exit 1
fi

# Convert the Docker tar file to a Singularity (Apptainer) SIF image
sudo apptainer build $IMAGE_NAME-image.sif docker-archive://$IMAGE_NAME-image.tar
if [ $? -ne 0 ]; then
    echo "Apptainer build failed. Exiting."
    exit 1
fi

# Copy the SIF image to the remote server
scp $IMAGE_NAME-image.sif $SERVER_ADDRESS:$SIF_FOLDER_LOCATION/$IMAGE_NAME-image.sif
if [ $? -ne 0 ]; then
    echo "SCP failed. Exiting."
    exit 1
fi

# Remove the Docker image and tar file to clean up space
sudo docker rmi $IMAGE_NAME-image:latest
rm $IMAGE_NAME-image.tar
rm $IMAGE_NAME-image.sif

# Prompt the user before running system prune
read -p "Do you want to run 'docker system prune' to clean up all unused Docker resources? [y/N]: " -r
echo  # new line
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo docker system prune -f  # If the user says 'y' or 'Y', perform the system prune
else
    echo "Skipping Docker system prune."
fi

echo "Script completed successfully, and Docker caches have been cleaned up."
