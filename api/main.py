from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import cv2
import multiprocessing
import uuid
from typing import List
from pymongo import MongoClient
from hsemotion_onnx.facial_emotions_demo_multi import process_video
from hsemotion_onnx.config import config

app = FastAPI()

client = MongoClient(config.MONGO_URL)
db = client.emotion_db

processes = {}
commands = {}

class EmotionRequest(BaseModel):
    video_type: str = "webcam"
    video_url: List[str] = ["rtsp urls"]
    skip_frame: int = 1
    timeout: int = 99999
    window_size: int = 5
    buffer_size: int = 10
    write_db: bool = False
    show: bool = False
    id: str = None

def check_video_stream(video_type: str, video_urls: list):
    for video_url in video_urls:
        cap = None
        if video_type == 'webcam':
            cap = cv2.VideoCapture(video_url)
        elif video_type == 'rtsp':
            if video_url is None:
                raise HTTPException(status_code=400, detail="RTSP stream requires video_url parameter")
            cap = cv2.VideoCapture(video_url)
        else:
            raise HTTPException(status_code=400, detail="Invalid video source")

        if not cap.isOpened():
            raise HTTPException(status_code=404, detail=f"Could not open {video_type} stream")

        cap.release()

def start_connection(video_type: str, video_urls: List[str], skip_frame: int, timeout: int, uuid: str, window_size: int, buffer_size: int, write_db: bool, show: bool, command_queue_list: list):
    processes = []
    if video_type == 'webcam':
        process = multiprocessing.Process(target=process_video, args=(None, 0, skip_frame, timeout, uuid, window_size, buffer_size, write_db, show, command_queue_list[0]))
        processes.append(process)
    else:
        for i, video_url in enumerate(video_urls):
            process = multiprocessing.Process(target=process_video, args=(video_url, None, skip_frame, timeout, uuid, window_size, buffer_size, write_db, show, command_queue_list[i]))
            processes.append(process)
    
    for process in processes:
        process.start()

    for process in processes:
        process.join()

@app.post("/init_connection")
async def init_connection_api(request: EmotionRequest, background_tasks: BackgroundTasks):
    try:
        id = request.id if request.id is not None else str(uuid.uuid4())
        video_urls = request.video_url if request.video_type != "webcam" else [0]
        
        command_queue = [multiprocessing.Queue() for _ in range(len(video_urls))]
        commands[id] = command_queue
        background_tasks.add_task(start_connection, request.video_type, video_urls, request.skip_frame, request.timeout, id, request.window_size, request.buffer_size, request.write_db, request.show, command_queue)
        
        return {"status": "success", "message": "Face detection process started", "uuid": id}
    except HTTPException as e:
        return {"status": "error", "message": str(e)}

@app.post("/face_detection")
async def face_detection_api(uuid: str):
    try:
        if uuid not in commands:
            raise HTTPException(status_code=404, detail="UUID not found")
        command_queue_list = commands[uuid]
        for command_queue in command_queue_list:
            command_queue.put('start')
        return {"status": "success", "message": "Face emotion detection started"}
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
