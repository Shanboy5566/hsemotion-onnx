import cv2
import mediapipe as mp
import numpy as np
import time
import multiprocessing
import uuid
import base64
import zlib
from typing import List

from pymongo import MongoClient
from collections import deque
import datetime
from .config import config

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

emotion_recognizer = HSEmotionRecognizer(model_name=config.MODEL_NAME)

# MongoDB client setup
client = MongoClient(config.MONGO_URL)
emotion_db = client.emotion_db
picture_db = client.picture_db

def write_to_db(table, buffer: List):
    if (len(buffer) == 0): return
    table.insert_many(buffer)
    buffer = []

def process_video(rtsp_url=None, 
                  webcam=None, 
                  skip_frame=config.SKIP_FRAME, 
                  window_size=config.WINDOW_SIZE, 
                  buffer_size=config.BUFFER_SIZE,
                  write_db=False,
                  write_picture=False,
                  show=False,
                  face_detection_confidence=0.25,
                  command_queue=None):

    cap = None
    if webcam is not None:
        cap = cv2.VideoCapture(webcam)
        webcam = "0"
    else:
        cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print(f"Error: Couldn't open {'webcam' if webcam else rtsp_url}")
        return

    print(f"Connection established for {'webcam' if webcam else rtsp_url}")

    timeout = 999999
    start_detection = False
    emotion_buffer = []
    picture_buffer = []

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=face_detection_confidence) as face_detection:
        start_time = time.time()
        frame_count = 0
        while cap.isOpened():
            if time.time() - start_time > timeout:
                print(f"Timeout reached for {'webcam' if webcam else rtsp_url}")
                start_detection = False
                timeout = 999999
                if show:
                    cv2.destroyAllWindows()
                if write_db:
                    emotion_db.emotions.insert_many(emotion_buffer)
                    emotion_buffer = []
                if write_picture:
                    picture_db.pictures.insert_many(picture_buffer)
                    picture_buffer = []

            success, image = cap.read()
            if not success:
                print(f"Ignoring empty frame from {'webcam' if webcam else rtsp_url}")
                continue

            frame_count += 1

            if not start_detection:
                if command_queue and not command_queue.empty():
                    command, timeout, uuid_str = command_queue.get()
                    print(f"Received command: {command}, timeout: {timeout}, uuid: {uuid_str}")
                    if command == 'start':
                        start_detection = True
                        start_time = time.time()
                        print(f"start: {start_time}, timeout: {timeout}")
                    else:
                        print(f"Invalid command received: {command}")
                        cap.release()
                        return
            else:
                if frame_count % skip_frame == 0:
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(image)

                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    if results.detections:
                        emotions = []
                        for detection in results.detections:
                            bboxC = detection.location_data.relative_bounding_box
                            ih, iw, _ = image.shape
                            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                         int(bboxC.width * iw), int(bboxC.height * ih)

                            x = max(0, x)
                            y = max(0, y)
                            w = min(iw - x, w)
                            h = min(ih - y, h)

                            face_img = image[y:y + h, x:x + w]

                            emotion, scores = emotion_recognizer.predict_emotions(face_img, logits=True)
                            emotion = np.argmax(scores)
                            emotion = emotion_recognizer.idx_to_class[emotion]
                            emotions.append(scores)

                            if show:
                                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                fontScale = 2
                                min_y = y if y >= 0 else 10
                                cv2.putText(image, f"{emotion}", (x, min_y), cv2.FONT_HERSHEY_PLAIN, fontScale=fontScale,
                                            color=(0, 255, 0), thickness=3)

                        resized_image = cv2.resize(image, (400, 300))
                        _, buffer = cv2.imencode('.jpg', resized_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                        jpg_as_text = zlib.compress(buffer)

                        for face_id, emotion in enumerate(emotions):
                            emotion_max_id = np.argmax(emotion)
                            emotion_max_class = emotion_recognizer.idx_to_class[emotion_max_id]
                            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

                            emotion_buffer.append({
                                "uuid": uuid_str,
                                "face_id": face_id,
                                "timestamp": timestamp,
                                "emotion": emotion_max_class,
                                "scores": emotion.tolist()
                            })

                        if write_picture:
                            picture_buffer.append({
                                "uuid": uuid_str,
                                "timestamp": timestamp,
                                "image": jpg_as_text
                            })

                        if write_db and len(emotion_buffer) >= buffer_size:
                            # Save to MongoDB
                            emotion_db.emotions.insert_many(emotion_buffer)
                            emotion_buffer = []

                        if write_picture and len(picture_buffer) >= buffer_size:
                            # Save to MongoDB
                            picture_db.pictures.insert_many(picture_buffer)
                            picture_buffer = []

                    if show:
                        cv2.imshow(f'Face Detection - {"webcam" if webcam else rtsp_url}', image)
                        if cv2.waitKey(5) & 0xFF == ord("q"):
                            break

    cap.release()
    if show:
        cv2.destroyAllWindows()

    if write_db:
        # Save remaining data to MongoDB
        if len(emotion_buffer) > 0:
            emotion_db.emotions.insert_many(emotion_buffer)

    if write_picture:
        if len(picture_buffer) >= buffer_size:
            # Save to MongoDB
            picture_db.pictures.insert_many(picture_buffer)
            picture_buffer = []