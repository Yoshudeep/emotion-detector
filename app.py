import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2

# Title
st.title("Live Camera Feed using Streamlit WebRTC")

# Custom video transformer (just displays frames)
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert the frame to BGR (OpenCV format)
        img = frame.to_ndarray(format="bgr24")

        # Optionally, you can add any processing here

        return img

# Start webcam stream
webrtc_streamer(key="live-camera", video_transformer_factory=VideoTransformer)
