import cv2
import numpy as np
import time
import zlib
import os
from pymongo import MongoClient
import datetime
import logging

from hsemotion_onnx.centerface import CenterFace
from hsemotion_onnx.config import config
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
from hsemotion_onnx.utils import sadness_normalization

# logger setup - write logs to file backend.log
logging.basicConfig(filename='backend.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

emotion_recognizer = HSEmotionRecognizer(model_name=config.MODEL_NAME)

# MongoDB client setup
client = MongoClient(config.MONGO_URL)
emotion_db = client.emotion_db

def write_picture_to_disk(picture_buffer):
    if (len(picture_buffer) == 0):
        return
    uuid_str = picture_buffer[0]["uuid"]
    device = picture_buffer[0]["device"]
    backup_dir = f'backup/{uuid_str}/{device}/images'

    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    for picture_id, picture in enumerate(picture_buffer):
        image = zlib.decompress(picture["image"])
        with open(f"./{backup_dir}/{uuid_str}-{picture_id}.jpg", 'wb') as img_file:
            img_file.write(image)

def check_connection(cap, video_source, max_retries=5, delay=1):
    logger.info(f"{video_source} | Checking connection")
    retry = 0
    while True:
        if cap.read()[0] == True:
            logger.info(f"{video_source} | Re-connected")
            return cap
        cap.release()
        logger.info(f"{video_source} | Retry {retry} | Connection lost. Retrying in {delay} seconds")
        time.sleep(delay)
        cap = cv2.VideoCapture(video_source)
        retry += 1

def process_video(
                rtsp_url=None, 
                webcam=None, 
                skip_frame=config.SKIP_FRAME, 
                window_size=config.WINDOW_SIZE, # deprecated
                buffer_size=config.BUFFER_SIZE,
                write_db=False,
                write_picture=False,
                show=False,
                face_detection_confidence=0.25, # deprecated
                image_zoom_factor=1.0,
                horizontal_splits=False,
                vertical_splits=False,
                command_queue=None
                ):

    """
    Process a video stream for facial emotion detection.

    Args:
        rtsp_url (str): The RTSP URL of the video stream. If provided, the `webcam` argument will be ignored.
        webcam (int or str): The index or name of the webcam to use. If provided, the `rtsp_url` argument will be ignored.
        skip_frame (int): The number of frames to skip between each face detection. Default is `config.SKIP_FRAME`.
        window_size (tuple): Deprecated argument. Not used in the function.
        buffer_size (int): The maximum number of emotion records to buffer before writing to the database. Default is `config.BUFFER_SIZE`.
        write_db (bool): Whether to write emotion records to the database. Default is `False`.
        write_picture (bool): Whether to write face images to disk. Default is `False`.
        show (bool): Whether to display the video stream with face detections. Default is `False`.
        face_detection_confidence (float): Deprecated argument. Not used in the function.
        image_zoom_factor (float): The zoom factor to apply to the input video frames. Default is `1.0`.
        horizontal_splits (bool): Whether to split the video frames horizontally. Default is `False`.
        vertical_splits (bool): Whether to split the video frames vertically. Default is `False`.
        command_queue (Queue): A queue for receiving commands to control the video processing. Default is `None`.

    Returns:
        None
    """

    cap = None
    is_webcam = False
    if webcam is not None:
        cap = cv2.VideoCapture(webcam)
        is_webcam = True
    else:
        cap = cv2.VideoCapture(rtsp_url)

    video_source = "webcam" if is_webcam else rtsp_url

    if not cap.isOpened():
        logger.info(f"Error: Couldn't open {video_source}")
        return

    logger.info(f"{video_source} | Connection established")

    timeout = 999999
    start_detection = False
    emotion_buffer = []
    picture_buffer = []

    # center face
    logger.info("Loading CenterFace model")
    centerface = CenterFace()

    start_time = time.time()
    frame_count = 0

    # We will open the video stream first
    while cap.isOpened():
        # Check if we need to stop the detection
        # When timeout is reached, we will stop the detection and write the remaining data to the database
        if time.time() - start_time > timeout:
            logger.info(f"{video_source} | Timeout reached")
            start_detection = False
            timeout = 999999
            if show:
                cv2.destroyAllWindows()

            # Write the remaining data to the database
            if write_db and len(emotion_buffer) > 0:
                emotion_db.emotions.insert_many(emotion_buffer)
                emotion_buffer = []

            # Only write the face images to disk if timeout is reached
            # and the write_picture flag is set
            if write_picture:
                write_picture_to_disk(picture_buffer)
                picture_buffer = []

        success, image = cap.read()
        if not success:
            logger.info(f"{video_source} | Ignoring empty frame from {video_source}")
            cap = check_connection(cap, video_source)
            if cap is None:
                logger.info(f"Error: Couldn't open {video_source}")
            continue

        frame_count += 1

        # Check if we need to start the detection        
        if not start_detection:
            # Only start the detection when the command_queue is not empty
            # There are three data in the queue: command, timeout, and uuid: 
            # 1. command can be 'start'
            # 2. timeout is the duration of the detection
            # 3. uuid is the unique identifier for the detection session
            if command_queue and not command_queue.empty():
                command, timeout, uuid_str = command_queue.get()
                logger.info(f"{video_source} | Received command: {command}, timeout: {timeout}, uuid: {uuid_str}")
                # cap = check_connection(cap, video_source)
                # if cap is None:
                #     logger.info(f"Error: Couldn't open {video_source}")
                #     return
                if command == 'start':
                    start_detection = True
                    start_time = time.time()
                    logger.info(f"{video_source} | start: {start_time}, timeout: {timeout}")
                else:
                    logger.info(f"{video_source} | Invalid command received: {command}")
                    cap.release()
                    return
        else:
            # success, image = cap.read()
            # if not success:
            #     logger.info(f"{video_source} | Ignoring empty frame from {video_source}")
            #     continue

            # frame_count += 1

            # Skip frames 
            if frame_count % skip_frame != 0:
                continue
            
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


            image = cv2.resize(image, (0, 0), fx=image_zoom_factor, fy=image_zoom_factor)
            height, width, _ = image.shape

            # If we need to split the image, we have to do it before the face detection
            # There are two types of splits: horizontal and vertical
            images = []
            if not horizontal_splits and not vertical_splits:
                images.append(image)

            if horizontal_splits:
                midpoint = height // 2
                top_image = image[:midpoint, :]
                bottom_image = image[midpoint:, :]
                images.append(top_image)
                images.append(bottom_image)
            
            if vertical_splits:
                midpoint = width // 2
                left_image = image[:, :midpoint]
                right_image = image[:, midpoint:]
                images.append(left_image)
                images.append(right_image)

            for split_id, image in enumerate(images):
                # Face detection
                dets, lms = centerface(image, height, width, threshold=0.35)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                emotions = []
                for detection in dets:
                    # Get the bounding box coordinates of the face
                    bboxC = detection[:4]
                    x, y, p, q = int(bboxC[0]), int(bboxC[1]), \
                                int(bboxC[2]), int(bboxC[3])
                    face_img = image[y:q, x:p]

                    # Emotion recognition
                    emotion, scores = emotion_recognizer.predict_emotions(face_img, logits=True)
                    scores = sadness_normalization(scores, sadness_id=6, offset=config.SADNESS_OFFSET)
                    emotion = np.argmax(scores)
                    emotion = emotion_recognizer.idx_to_class[emotion]
                    emotions.append(scores)

                    cv2.rectangle(image, (x, y), (p, q), (0, 255, 0), 2)
                    fontScale = 2
                    min_y = y if y >= 0 else 10
                    cv2.putText(image, f"{emotion}", (x, min_y), cv2.FONT_HERSHEY_PLAIN, fontScale=fontScale,
                                color=(0, 255, 0), thickness=3)

                # Resize the image for saving
                timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                resized_image = cv2.resize(image, (800, 600))
                _, buffer = cv2.imencode('.jpg', resized_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                jpg_as_text = zlib.compress(buffer)

                # Save the emotion data
                for face_id, emotion in enumerate(emotions):
                    emotion_max_id = np.argmax(emotion)
                    emotion_max_class = emotion_recognizer.idx_to_class[emotion_max_id]

                    emotion_buffer.append({
                        "uuid": uuid_str,
                        "device": video_source.split('@')[-1].split(':')[0],
                        "face_id": face_id,
                        "split_id": split_id,
                        "timestamp": timestamp,
                        "emotion": emotion_max_class,
                        "scores": emotion.tolist()
                    })

                    # Write to MongoDB if buffer is full
                    if write_db and len(emotion_buffer) >= buffer_size:
                        # Save to MongoDB
                        emotion_db.emotions.insert_many(emotion_buffer)
                        emotion_buffer = []
                
                # Write the remaining data to the database
                if write_db and len(emotion_buffer) > 0:
                    # Save to MongoDB
                    emotion_db.emotions.insert_many(emotion_buffer)
                    emotion_buffer = []

                # Save the face image
                if write_picture:
                    picture_buffer.append({
                        "uuid": uuid_str,
                        "device": video_source.split('@')[-1].split(':')[0],
                        "split_id": split_id,
                        "timestamp": timestamp,
                        "image": jpg_as_text
                    })

                if show:
                    cv2.imshow(f'Face Detection - {"webcam" if is_webcam else rtsp_url} - {split_id}', image)
                    if cv2.waitKey(5) & 0xFF == ord("q"):
                        break

    cap.release()
    if show:
        cv2.destroyAllWindows()

    if write_db:
        # Save remaining data to MongoDB
        if len(emotion_buffer) > 0:
            emotion_db.emotions.insert_many(emotion_buffer)