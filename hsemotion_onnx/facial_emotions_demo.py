import cv2
import numpy as np
import argparse
import time
from collections import deque
import logging
from screeninfo import get_monitors

from hsemotion_onnx.centerface import CenterFace
from hsemotion_onnx.config import config
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
from hsemotion_onnx.utils import sadness_normalization, emotion_to_branch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

model_name='enet_b0_8_best_vgaf'
#model_name='enet_b0_8_va_mtl'

emotion_recognizer = HSEmotionRecognizer(model_name=model_name)

maxlen=15 #51
recent_scores=deque(maxlen=maxlen)

monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

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

def get_color(emotion: str):
    # Determine color based on emotion
    if emotion == 'Neutral':
        return (0, 255, 255)  # Yellow
    elif emotion == 'Positive':
        return (0, 255, 0)    # Green
    elif emotion == 'Negative':
        return (0, 0, 255)    # Red

def process_video(video_sources, skip_frame=1, timeout=None):
    # Load CenterFace model
    print("Loading CenterFace model")
    centerface = CenterFace()

    current_source_idx = 0
    cap = None
    start_time = time.time()

    while True:
        video_source = video_sources[current_source_idx]
        logger.info(f"video_source: {video_source}")

        # Open the video source
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            logger.error(f"Error: Could not open {video_source} stream.")
            current_source_idx = (current_source_idx + 1) % len(video_sources)
            continue

        start = time.time()
        frame_count = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                logger.info(f"{video_source} | Ignoring empty frame from {video_source}")
                cap = check_connection(cap, video_source)
                if cap is None:
                    logger.info(f"Error: Couldn't open {video_source}")
                continue

            # Switch video source every x seconds
            if time.time() - start_time > config.SWITCH_VIDEO_SOURCE_INTERVAL:
                start_time = time.time()
                current_source_idx = (current_source_idx + 1) % len(video_sources)
                cap.release()
                logger.info(f"Switch")
                break

            # Check for timeout
            if time.time() - start > timeout:
                logger.info(f"Timeout reached. Stopping video processing.")
                break

            frame_count += 1
            if frame_count % skip_frame != 0:
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = image.shape

            # Face detection
            dets, lms = centerface(image, height, width, threshold=0.35)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            for detection in dets:
                bboxC = detection[:4]
                x, y, p, q = int(bboxC[0]), int(bboxC[1]), int(bboxC[2]), int(bboxC[3])
                face_img = image[y:q, x:p]

                # Emotion recognition
                emotion, scores = emotion_recognizer.predict_emotions(face_img, logits=True)
                scores = sadness_normalization(scores, sadness_id=6, offset=config.SADNESS_OFFSET)
                emotion = np.argmax(scores)
                emotion = emotion_recognizer.idx_to_class[emotion]
                color = (0, 255, 0)    # Green
                if config.EMOTION_TO_BRANCH:
                    emotion, color = emotion_to_branch(emotion)

                cv2.rectangle(image, (x, y), (p, q), color, 2)
                fontScale = 2
                min_y = y if y >= 0 else 10
                cv2.putText(image, f"{emotion}", (x, min_y), cv2.FONT_HERSHEY_PLAIN, fontScale=fontScale,
                            color=color, thickness=3)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

            cv2.namedWindow('MediaPipe Face Detection', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('MediaPipe Face Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            resized_image = cv2.resize(image, (screen_width, screen_height))
            cv2.imshow('MediaPipe Face Detection', resized_image)

        cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process video with emotion detection.')
    parser.add_argument('--file', help='Path to video file')
    parser.add_argument('--rtsp-urls', nargs='+', type=str, help='RTSP URLs')
    parser.add_argument('--skip-frame', type=int, default=1, help='Number of frames to skip before processing each frame')
    parser.add_argument('--timeout', type=int, default=9999999, help='Timeout in seconds to stop video processing')
    args = parser.parse_args()

    if args.file:
        process_video([args.file], args.skip_frame, args.timeout)
    elif args.rtsp_urls:
        process_video(args.rtsp_urls, args.skip_frame, args.timeout)
    else:
        process_video([0], args.skip_frame, args.timeout)
