import streamlit as st
import cv2
import numpy as np
import time
import mediapipe as mp
import gdown
import os
import asyncio
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av  # Make sure to import av for VideoFrame usage

# Suppress specific asyncio sendto errors (transport NoneType)
def handle_asyncio_exception(loop, context):
    msg = context.get("exception", context.get("message"))
    if msg and "sendto" in str(msg):
        # Suppress this specific error related to sendto NoneType
        return
    # For other exceptions, log normally
    logging.error(f"Caught asyncio exception: {context}")

def set_asyncio_exception_handler():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.set_exception_handler(handle_asyncio_exception)

set_asyncio_exception_handler()

# Config
IMG_SIZE = 48
EAR_THRESHOLD = 0.21
CONSEC_FRAMES = 3
emotion_labels = ['angry', 'fear', 'happy', 'neutral', 'sad']
EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

st.set_page_config(page_title="Emotion Detection", layout="centered")

# Title
st.markdown("""
    <style>
        .title { text-align: center; font-size: 2.5rem; font-weight: bold; color: #4CAF50; }
        .subheader { text-align: center; font-size: 1.2rem; color: #777; }
    </style>
""", unsafe_allow_html=True)
st.markdown("<div class='title'>Real-Time Emotion Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>With Blink & Motion Spoof Detection</div>", unsafe_allow_html=True)
st.markdown("---")

# Model loader
@st.cache_resource
def load_emotion_model():
    model_path = "emotion_model.h5"
    if not os.path.exists(model_path):
        gdown.download("https://drive.google.com/uc?id=1Yl3TbQiQ2MKAQjWBiUF2SjNCQ4hyh2bD", model_path, quiet=False)
    return load_model(model_path)

model = load_emotion_model()


# EAR Calculator
def calculate_ear(landmarks, w, h):
    from math import dist
    p1 = (int(landmarks[0].x * w), int(landmarks[0].y * h))
    p2 = (int(landmarks[1].x * w), int(landmarks[1].y * h))
    p3 = (int(landmarks[2].x * w), int(landmarks[2].y * h))
    p4 = (int(landmarks[3].x * w), int(landmarks[3].y * h))
    p5 = (int(landmarks[4].x * w), int(landmarks[4].y * h))
    p6 = (int(landmarks[5].x * w), int(landmarks[5].y * h))
    return (dist(p2, p6) + dist(p3, p5)) / (2.0 * dist(p1, p4))


# Video Processor Class
class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self, model):
        self.model = model
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.blink_counter = 0
        self.frame_counter = 0
        self.prev_face_coords = None
        self.last_movement_time = time.time()
        self.last_blink_time = time.time()

    def recv(self, frame):
        try:
            image = frame.to_ndarray(format="bgr24")
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = image.shape[:2]

            result = self.face_mesh.process(rgb)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            face_detected = False
            is_live_face = False

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0]
                eye_points = [landmarks.landmark[i] for i in EYE_LANDMARKS]
                ear = calculate_ear(eye_points, w, h)

                if ear < EAR_THRESHOLD:
                    self.frame_counter += 1
                else:
                    if self.frame_counter >= CONSEC_FRAMES:
                        self.blink_counter += 1
                        self.last_blink_time = time.time()
                    self.frame_counter = 0
                face_detected = True

            if len(faces) > 0:
                (x, y, w_box, h_box) = faces[0]
                if self.prev_face_coords:
                    dx = abs(x - self.prev_face_coords[0])
                    dy = abs(y - self.prev_face_coords[1])
                    if dx > 5 or dy > 5:
                        self.last_movement_time = time.time()
                self.prev_face_coords = (x, y)

            time_since_blink = time.time() - self.last_blink_time
            time_since_move = time.time() - self.last_movement_time

            is_live_face = (time_since_blink < 3) and (time_since_move < 3)

            # Overlay
            if is_live_face:
                cv2.putText(image, "LIVE FACE", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            else:
                cv2.putText(image, "SPOOF DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Emotion prediction
            if face_detected and is_live_face and len(faces) > 0:
                (x, y, w_box, h_box) = faces[0]
                face_roi = gray[y:y + h_box, x:x + w_box]
                face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
                face_resized = np.expand_dims(face_resized, axis=-1)
                face_array = img_to_array(face_resized)
                face_array = np.expand_dims(face_array, axis=0) / 255.0

                preds = self.model.predict(face_array, verbose=0)
                emotion = emotion_labels[np.argmax(preds)]

                cv2.rectangle(image, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                cv2.putText(image, f'{emotion.upper()}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            return av.VideoFrame.from_ndarray(image, format="bgr24")

        except Exception as e:
            # Log error and skip frame (return input frame unmodified)
            print(f"Error processing frame: {e}")
            return frame


# Factory
def processor_factory():
    return EmotionVideoProcessor(model)


# Run webrtc streamer
webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=processor_factory,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
