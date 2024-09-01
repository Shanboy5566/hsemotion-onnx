#/bin/bash

PROJECT="/Users/gotop_mini_m2/hsemotion-onnx"
# PROJECT="/Users/shanboy/Documents/project/hsemotion-onnx"
RTSP_URL="rtsp://admin:kris0226@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0"
# RTSP_URL="rtsp://97.68.104.34/axis-media/media.amp"

export SADNESS_OFFSET=-0.55
export EMOTION_TO_BRANCH=1

# Function to clean up background processes
cleanup() {
    echo "Cleaning up..."
    pkill -f "python"      # Kills the Python process
    echo "Done."
}

# Ensure cleanup is called in the beginning
cleanup

cd $PROJECT
source py39/bin/activate \
&& cd hsemotion_onnx \
&& python facial_emotions_demo.py --skip-frame 10 --rtsp-url $RTSP_URL
# && python facial_emotions_demo.py 

cleanup