import cv2
import mediapipe as mp
import numpy as np
import argparse
import time
import multiprocessing
import uuid
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
db = client.emotion_db

def process_video(rtsp_url=None, 
                  webcam=None, 
                  skip_frame=config.SKIP_FRAME, 
                  timeout=config.TIMEOUT, 
                  uuid_str=None, 
                  window_size=config.WINDOW_SIZE, 
                  buffer_size=config.BUFFER_SIZE,
                  write_db=False, 
                  show=False):
    """
    Process a video stream to detect facial emotions.

    Args:
        rtsp_url (str, optional): The RTSP URL of the video stream. Defaults to None.
        webcam (int, optional): The index of the webcam to use. Defaults to None.
        skip_frame (int, optional): The number of frames to skip between detections. Defaults to config.SKIP_FRAME.
        timeout (int, optional): The maximum time in seconds to process the video stream. Defaults to config.TIMEOUT.
        uuid_str (str, optional): The UUID string to identify the video stream. Defaults to None.
        window_size (int, optional): The size of the emotion queue window. Defaults to config.WINDOW_SIZE.
        buffer_size (int, optional): The size of the emotion buffer. Defaults to config.BUFFER_SIZE.
        write_db (bool, optional): Whether to write the emotion data to a database. Defaults to False.
        show (bool, optional): Whether to display the video stream with bounding boxes and emotion labels. Defaults to False.
    """
     
    if uuid_str is None:
        uuid_str = str(uuid.uuid4())

    cap = None
    if webcam is not None:
        cap = cv2.VideoCapture(webcam)
    else:
        cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print(f"Error: Couldn't open {'webcam' if webcam else rtsp_url}")
        return

    emotion_queue = deque(maxlen=window_size)
    emotion_buffer = []
    
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.25) as face_detection:
        start_time = time.time()
        frame_count = 0
        while cap.isOpened():
            if time.time() - start_time > timeout:
                print(f"Timeout reached for {'webcam' if webcam else rtsp_url}")
                break

            success, image = cap.read()
            if not success:
                print(f"Ignoring empty frame from {'webcam' if webcam else rtsp_url}")
                continue

            frame_count += 1
            if frame_count % skip_frame != 0:
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detections:
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
                    emotion_queue.append(scores)

                    if show:
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        fontScale = 2
                        min_y = y if y >= 0 else 10
                        cv2.putText(image, f"{emotion}", (x, min_y), cv2.FONT_HERSHEY_PLAIN, fontScale=fontScale,
                                    color=(0, 255, 0), thickness=3)

            if len(emotion_queue) == window_size:
                avg_scores = np.mean(emotion_queue, axis=0)
                avg_emotion = np.argmax(avg_scores)
                avg_emotion = emotion_recognizer.idx_to_class[avg_emotion]
                timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

                emotion_buffer.append({
                    "uuid": uuid_str,
                    "timestamp": timestamp,
                    "emotion": avg_emotion,
                    "scores": avg_scores.tolist()
                })

                if write_db and len(emotion_buffer) >= buffer_size:
                    # Save to MongoDB
                    db.emotions.insert_many(emotion_buffer)
                    emotion_buffer = []

            if show:
                cv2.imshow(f'Face Detection - {"webcam" if webcam else rtsp_url}', image)
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
    cap.release()
    if show:
        cv2.destroyAllWindows()
    
    if write_db and emotion_buffer:
        # Save remaining data to MongoDB
        db.emotions.insert_many(emotion_buffer)