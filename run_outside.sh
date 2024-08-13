#/bin/bash

PROJECT="/Users/mac/Documents/hsemotion-onnx"
RTSP_URL="rtsp://admin:kris0226@192.168.1.101:554/cam/realmonitor?channel=1&subtype=0"

cd $PROJECT
source py312/bin/activate \
&& cd hsemotion_onnx \
&& python facial_emotions_demo.py --skip-frame 13 --rtsp-url $RTSP_URL