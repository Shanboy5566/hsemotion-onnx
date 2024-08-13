#!/bin/bash

PROJECT="/Users/mac/Documents/hsemotion-onnx"
VIDEO_PATH="/Users/mac/Documents/mpv_ctrl/0802.mov"

# Orbstack
echo "Starting Orbstack server..."
orb start

# MongoDB
echo "Starting MongoDB..."
docker restart mongodb

# FastAPI
echo "Starting FastAPI server..."
cd $PROJECT
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
read -p "Press any key to continue..."

# A & B & C