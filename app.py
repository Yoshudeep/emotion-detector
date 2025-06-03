from streamlit_webrtc import webrtc_streamer

webrtc_streamer(
    key="test",
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,
)
