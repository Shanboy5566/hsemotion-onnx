from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import cv2
import multiprocessing
import uuid
import base64
import os
import json
from typing import List
from pymongo import MongoClient
from hsemotion_onnx.facial_emotions_demo_multi import process_video
from hsemotion_onnx.config import config

app = FastAPI()

client = MongoClient(config.MONGO_URL)
db = client.emotion_db

id = str(uuid.uuid4())
processes = {}
commands = {}

class InitRequest(BaseModel):
    video_type: str = "webcam"
    video_url: List[str] = ["rtsp urls"]
    skip_frame: int = 1
    window_size: int = 1
    buffer_size: int = 10
    write_db: bool = False
    show: bool = False

class FaceRequest(BaseModel):
    uuid: str = "uuid"
    timeout: int = 999999

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

def start_connection(video_type: str, video_urls: List[str], skip_frame: int, window_size: int, buffer_size: int, write_db: bool, show: bool, command_queue_list: list):
    processes = []
    if video_type == 'webcam':
        process = multiprocessing.Process(target=process_video, args=(None, 0, skip_frame, window_size, buffer_size, write_db, show, command_queue_list[0]))
        processes.append(process)
    else:
        for i, video_url in enumerate(video_urls):
            process = multiprocessing.Process(target=process_video, args=(video_url, None, skip_frame, window_size, buffer_size, write_db, show, command_queue_list[i]))
            processes.append(process)
    
    for process in processes:
        process.start()

    for process in processes:
        process.join()

@app.post("/init_connection")
async def init_connection_api(request: InitRequest, background_tasks: BackgroundTasks):
    try:
        video_urls = request.video_url if request.video_type != "webcam" else [0]
        
        command_queue = [multiprocessing.Queue() for _ in range(len(video_urls))]
        commands[id] = command_queue
        background_tasks.add_task(start_connection, request.video_type, video_urls, request.skip_frame, request.window_size, request.buffer_size, request.write_db, request.show, command_queue)
        
        return {"status": "success", "message": "Face detection process started", "uuid": id}
    except HTTPException as e:
        return {"status": "error", "message": str(e)}

@app.post("/face_detection")
async def face_detection_api(request: FaceRequest):
    try:
        command_queue_list = commands[id]
        for command_queue in command_queue_list:
            command_queue.put(['start', request.timeout, request.uuid])
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
        return {"status": "error", "message": str(e), "results": []}

@app.post("/backup_to_disk/{uuid_str}")
async def backup_to_disk_api(uuid_str: str):
    try:
        # Create backup directory if it doesn't exist
        backup_dir = f'backup/{uuid_str}'
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)

        # Backup emotions
        emotions = list(db.emotions.find({"uuid": uuid_str}, {"_id": 0}))  # Exclude _id field
        with open(f'{backup_dir}/emotions.json', 'w') as f:
            json.dump(emotions, f, ensure_ascii=False, indent=4)

        # Backup pictures
        pictures = list(db.pictures.find({"uuid": uuid_str}))
        for i, picture in enumerate(pictures):
            image_data = base64.b64decode(picture["image"])
            with open(f'{backup_dir}/image_{i}.jpg', 'wb') as img_file:
                img_file.write(image_data)

        # Delete from DB
        db.emotions.delete_many({"uuid": uuid_str})
        db.pictures.delete_many({"uuid": uuid_str})

        return {"status": "success", "message": "Backup completed and data removed from database"}
    except Exception as e:
        return {"status": "error", "message": str(e)}