from fastapi import FastAPI, HTTPException
import cv2

from hsemotion_onnx.facial_emotions_demo import process_video

app = FastAPI()

# Function to check if webcam or RTSP stream connection is valid
def check_video_stream(video_type: str, video_url: str):
    cap = None
    if video_type == 'webcam':
        cap = cv2.VideoCapture(0)
    elif video_type == 'rtsp':
        if video_url is None:
            raise HTTPException(status_code=400, detail="RTSP stream requires video_url parameter")
        cap = cv2.VideoCapture(video_url)
    else:
        raise HTTPException(status_code=400, detail="Invalid video source")

    if not cap.isOpened():
        raise HTTPException(status_code=404, detail=f"Could not open {video_type} stream")

    cap.release()

@app.get("/check_video_stream")
async def check_video_stream_api(video_type: str, video_url: str = None):
    try:
        check_video_stream(video_type, video_url)
        return {"status": "success", "message": f"{video_type} stream is valid"}
    except HTTPException as e:
        return {"status": "error", "message": str(e)}

@app.get("/face_detection")
async def face_detection_api(video_type: str, video_url: str = None, skip_frame: int = 1):
    try:
        check_video_stream(video_type, video_url)
        process_video(video_type, video_url, skip_frame)
        return {"status": "success", "message": "Face detection completed"}
    except HTTPException as e:
        return {"status": "error", "message": str(e)}
