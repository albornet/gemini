#!/bin/bash

# Define variables
IMAGE_NAME="gemini"
SERVER_ADDRESS="borneta@login2.baobab.hpc.unige.ch"
SIF_FOLDER_LOCATION="/home/users/b/borneta/sif"

# Build the Docker image
docker build -t $IMAGE_NAME-image:latest .
if [ $? -ne 0 ]; then
    echo "Docker build failed. Exiting."
    exit 1
fi

# Save the Docker image to a tar file
docker save $IMAGE_NAME-image:latest > $IMAGE_NAME-image.tar
if [ $? -ne 0 ]; then
    echo "Docker save failed. Exiting."
    exit 1
fi

# # Prompt the user to clean docker cache (my computer disk is small...)
# read -p "Do you want to clean up all unused Docker resources? [y/N]: " -r
# echo  # new line
# if [[ $REPLY =~ ^[Yy]$ ]]; then
#     docker image prune -f
#     docker container prune -f
#     docker builder prune -f
#     docker system prune -f
# else
#     echo "Skipping docker cache cleaning"
# fi

# Convert the Docker tar file to a Singularity (Apptainer) SIF image
apptainer build $IMAGE_NAME-image.sif docker-archive://$IMAGE_NAME-image.tar
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
docker rmi $IMAGE_NAME-image:latest
rm $IMAGE_NAME-image.tar
rm $IMAGE_NAME-image.sif

echo "Script completed successfully, and Docker caches have been cleaned up."
