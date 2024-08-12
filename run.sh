#!/bin/bash

PROJECT="/Users/shanboy/Documents/project/hsemotion-onnx"

# Orbstack
echo "Starting Orbstack server..."
orb start

# MongoDB
echo "Starting MongoDB..."
docker restart mongodb

# FastAPI
echo "Starting FastAPI server..."
cd $PROJECT
source venv/bin/activate && cd api && uvicorn main:app --reload

# Wait for all processes to finish
read -p "Press any key to continue..."