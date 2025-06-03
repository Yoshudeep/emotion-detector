import streamlit as st
import cv2
import numpy as np
import time
import mediapipe as mp
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from math import dist
from streamlit_webrtc import webrtc_streamer
import av

st.set_page_config(page_title="Emotion Detection", layout="centered")

@st.cache_resource
def load_emotion_model():
    model_path = "emotion_model.h5"
    if not os.path.exists(model_path):
        file_id = "1Yl3TbQiQ2MKAQjWBiUF2SjNCQ4hyh2bD"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    return load_model(model_path)

EAR_THRESHOLD = 0.21
CONSEC_FRAMES = 3
IMG_SIZE = 48
EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
emotion_labels = ['angry', 'fear', 'happy', 'neutral', 'sad']

model = load_emotion_model()

st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            color: #4CAF50;
        }
        .subheader {
            text-align: center;
            font-size: 1.2rem;
            color: #777;
        }
        .status {
            font-size: 1.2rem;
            padding: 0.5rem;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Real-Time Emotion Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>With Blink & Motion Spoof Detection</div>", unsafe_allow_html=True)
st.markdown("---")

class EmotionDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.blink_counter = 0
        self.frame_counter = 0
        self.prev_face_coords = None
        self.last_movement_time = time.time()
        self.last_blink_time = time.time()

    def calculate_ear(self, landmarks, w, h):
        p1 = (int(landmarks[0].x * w), int(landmarks[0].y * h))
        p2 = (int(landmarks[1].x * w), int(landmarks[1].y * h))
        p3 = (int(landmarks[2].x * w), int(landmarks[2].y * h))
        p4 = (int(landmarks[3].x * w), int(landmarks[3].y * h))
        p5 = (int(landmarks[4].x * w), int(landmarks[4].y * h))
        p6 = (int(landmarks[5].x * w), int(landmarks[5].y * h))
        return (dist(p2, p6) + dist(p3, p5)) / (2.0 * dist(p1, p4))

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        result = self.face_mesh.process(rgb)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        face_detected = False
        is_live_face = False
        emotion = None

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0]
            eye_points = [landmarks.landmark[i] for i in EYE_LANDMARKS]
            ear = self.calculate_ear(eye_points, w, h)

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

        is_live_face = (time_since_blink < 3) and (time_since_move < 3)  # BOTH blink AND move within 3 sec

        # Draw live/spoof text
        cv2.putText(
            rgb,
            "LIVE FACE" if is_live_face else "SPOOF DETECTED",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0) if is_live_face else (0, 0, 255),
            3
        )

        # Draw blink count for debug
        cv2.putText(
            rgb,
            f'Blink count: {self.blink_counter}',
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        if face_detected and is_live_face and len(faces) > 0:
            (x, y, w_box, h_box) = faces[0]
            face_roi = gray[y:y + h_box, x:x + w_box]
            face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
            face_resized = np.expand_dims(face_resized, axis=-1)
            face_array = img_to_array(face_resized)
            face_array = np.expand_dims(face_array, axis=0) / 255.0

            preds = model.predict(face_array, verbose=0)
            emotion = emotion_labels[np.argmax(preds)]

            cv2.rectangle(rgb, (x, y), (x + w_box, y + h_box), (0, 255, 0), 3)
            cv2.putText(
                rgb,
                f'{emotion.upper()}',
                (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3
            )

        # Resize frame to bigger size for better view
        DESIRED_WIDTH, DESIRED_HEIGHT = 640, 480
        rgb_resized = cv2.resize(rgb, (DESIRED_WIDTH, DESIRED_HEIGHT))

        return av.VideoFrame.from_ndarray(cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR), format="bgr24")

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    if not hasattr(video_frame_callback, "detector"):
        video_frame_callback.detector = EmotionDetector()
    return video_frame_callback.detector.recv(frame)

webrtc_streamer(
    key="emotion-detection",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)
