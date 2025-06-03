import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2

# RTC Configuration with public STUN server
rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Title
st.title("Live Camera Feed with STUN Server (WebRTC)")

# Video transformer class
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        return img

# Stream webcam
webrtc_streamer(
    key="live-camera",
    video_transformer_factory=VideoTransformer,
    rtc_configuration=rtc_config
)
