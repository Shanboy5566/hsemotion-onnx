#!/bin/bash

PROJECT="/Users/mac/Documents/hsemotion-onnx"

# Orbstack
echo "Starting Orbstack server..."
orb start

# MongoDB
echo "Starting MongoDB..."
docker restart mongodb

# FastAPI
echo "Starting FastAPI server..."
cd $PROJECT
source py312/bin/activate && cd api && uvicorn main:app --reload

# Wait for all processes to finish
read -p "Press any key to continue..."