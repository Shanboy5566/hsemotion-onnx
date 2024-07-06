import cv2
import mediapipe as mp
import numpy as np
import argparse
import time
from collections import deque
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

model_name='enet_b0_8_best_vgaf'
#model_name='enet_b0_8_va_mtl'

emotion_recognizer = HSEmotionRecognizer(model_name=model_name)

maxlen=15 #51
recent_scores=deque(maxlen=maxlen)

def process_video(video_source, parameter, skip_frame=1, timeout=None):
    if video_source == 'webcam':
        cap = cv2.VideoCapture(0)
    elif video_source == 'file':
        cap = cv2.VideoCapture(parameter)
    elif video_source == 'rtsp':
        cap = cv2.VideoCapture(parameter)
    else:
        logger.error("Error: Invalid video source.")
        return

    if not cap.isOpened():
        logger.error(f"Error: Could not open {video_source} stream.")
        return

    start = time.time()
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.25) as face_detection:
        frame_count = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                logger.error("Ignoring empty camera frame.")
                continue

            if time.time() - start > timeout:
                logger.info(f"Timeout reached. Stopping video processing.")
                break

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

                    # check image boundary
                    x = max(0, x)
                    y = max(0, y)
                    w = min(iw - x, w)
                    h = min(ih - y, h)
                    
                    # crop face image
                    face_img = image[y:y+h, x:x+w]

                    emotion, scores = emotion_recognizer.predict_emotions(face_img, logits=True)
                    recent_scores.append(scores)
                    recent_scores_avg=np.mean(recent_scores,axis=0)

                    emotion = np.argmax(scores)
                    emotion = emotion_recognizer.idx_to_class[emotion]

                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    fontScale = 2
                    min_y = y if y >= 0 else 10
                    cv2.putText(image, f"{emotion}", (x, min_y), cv2.FONT_HERSHEY_PLAIN, fontScale=fontScale, color=(0, 255, 0), thickness=3)
            
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break
            cv2.imshow('MediaPipe Face Detection', image)

    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process video with emotion detection.')
    parser.add_argument('--file', help='Path to video file')
    parser.add_argument('--rtsp-url', help='RTSP URL')
    parser.add_argument('--skip-frame', type=int, default=1, help='Number of frames to skip before processing each frame')
    parser.add_argument('--timeout', type=int, default=9999999, help='Timeout in seconds to stop video processing')
    args = parser.parse_args()

    if args.file:
        process_video('file', args.file, args.skip_frame, args.timeout)
    elif args.rtsp_url:
        process_video('rtsp', args.rtsp_url, args.skip_frame, args.timeout)
    else:
        process_video('webcam', None, args.skip_frame, args.timeout)
