import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2

# RTC Configuration with STUN and TURN servers
rtc_config = RTCConfiguration({
    "iceServers": [
        # Google STUN server (free)
        {"urls": "stun:stun.l.google.com:19302"},
        
        # Public TURN server (free-tier with metered usage)
        {
            "urls": [
                "turn:openrelay.metered.ca:80",
                "turn:openrelay.metered.ca:443",
                "turn:openrelay.metered.ca:443?transport=tcp"
            ],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        }
    ]
})

# Streamlit title
st.title("Live Camera Feed with STUN + TURN (WebRTC)")

# Define custom transformer
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        return img

# Start webcam streamer
webrtc_streamer(
    key="live-camera",
    video_transformer_factory=VideoTransformer,
    rtc_configuration=rtc_config
)
