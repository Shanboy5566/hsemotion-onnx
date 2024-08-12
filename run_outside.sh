#/bin/bash

PROJECT="/Users/shanboy/Documents/project/hsemotion-onnx"

cd $PROJECT
source venv/bin/activate && cd hsemotion_onnx && python facial_emotions_demo.py --skip-frame 13 # --rtsp-url rtsp://