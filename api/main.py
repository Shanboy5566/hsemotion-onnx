from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import cv2
import multiprocessing
import uuid
from pymongo import MongoClient
from hsemotion_onnx.facial_emotions_demo_multi import process_video
from hsemotion_onnx.config import config

app = FastAPI()

client = MongoClient(config.MONGO_URL)
db = client.emotion_db

class EmotionRequest(BaseModel):
    video_type: str = "webcam"
    video_url: str = None
    skip_frame: int = 1
    timeout: int = 99999
    window_size: int = 5
    write_db: bool = False
    show: bool = False
    id: str = None

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

def start_face_detection(request: EmotionRequest, uuid: str):
    check_video_stream(request.video_type, request.video_url)
    if request.video_type == 'webcam':
        process = multiprocessing.Process(target=process_video, args=(None, 0, request.skip_frame, request.timeout, uuid, request.window_size, request.write_db, request.show))
    else:
        process = multiprocessing.Process(target=process_video, args=(request.video_url, None, request.skip_frame, request.timeout, uuid, request.window_size, request.write_db, request.show))
    process.start()

@app.post("/face_detection")
async def face_detection_api(request: EmotionRequest, background_tasks: BackgroundTasks):
    try:
        id = request.id if request.id is not None else str(uuid.uuid4())
        background_tasks.add_task(start_face_detection, request, id)
        return {"status": "success", "message": "Face detection process started", "uuid": id}
    except HTTPException as e:
        return {"status": "error", "message": str(e)}

@app.get("/get_emotion_result/{uuid_str}")
async def get_emotion_result(uuid_str: str):
    try:
        results = list(db.emotions.find({"uuid": uuid_str}, {"_id": 0}))
        if not results:
            raise HTTPException(status_code=404, detail="No results found for given UUID")
        return {"status": "success", "results": results}
    except HTTPException as e:
        return {"status": "error", "message": str(e)}
