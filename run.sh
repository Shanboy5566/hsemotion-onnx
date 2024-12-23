#!/bin/bash

PROJECT="/Users/mac/Documents/hsemotion-onnx"
VIDEO_PATH="/Users/mac/Documents/mpv_ctrl/0802.mov"

# PROJECT="/Users/shanboy/Documents/project/hsemotion-onnx"
# VIDEO_PATH="/Users/mac/Documents/mpv_ctrl/0802.mov"

export SADNESS_OFFSET=0.0
# export MODEL_NAME="enet_b2_8_best"

# Function to clean up background processes
cleanup() {
    echo "Cleaning up..."
    pkill -f "uvicorn"  # Kills the FastAPI process
    pkill -f "mpv"      # Kills the MPV process
    pkill -f "sleep"    # Kills the sleep process
    pkill -f "python"      # Kills the Python process
    echo "Done."
}

# Ensure cleanup is called in the beginning
cleanup

# Set trap to catch SIGINT (Ctrl + C) and run cleanup function
trap cleanup SIGINT

# Docker desktop
cd $PROJECT
echo "Starting Docker desktop..."
open -a Docker

while ! docker info > /dev/null 2>&1; do
  echo "Waiting for Docker to start..."
  sleep 2
done

docker compose up -d

# FastAPI
echo "Starting FastAPI server..."
source py312/bin/activate || exit 1 \
&& cd api \
&& uvicorn main:app --reload >> backend.log &

sleep 5

# MPV Contral
echo "Start MPV..."
mpv $VIDEO_PATH --hwdec=videotoolbox --fs --pause --input-ipc-server=/tmp/mpv-socket &
cd /Users/mac/Documents/mpv_ctrl/ && \
source venv/bin/activate && \
python3 mpv-controller.py

# Wait for all processes to finish
# read -p "Press any key to continue..."
cleanup  # Ensure cleanup is called even after the read command
exit 0