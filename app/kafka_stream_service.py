import logging
from app.streaming_services.VideoFrame import VideoStreamer

def stream_to_kafka():
    """Main function to start video streaming"""
    logging.basicConfig(level=logging.INFO)
    try:
        streamer = VideoStreamer(camera_index=0, delay=0.5)
        streamer.start()
    except KeyboardInterrupt:
        print("Streaming stopped by user")
    except Exception as e:
        logging.error(f"Error in main: {e}")

if __name__ == "__main__":
    stream_to_kafka()
