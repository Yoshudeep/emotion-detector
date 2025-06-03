from streamlit_webrtc import webrtc_streamer

webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=processor_factory,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    async_processing=True,
)
