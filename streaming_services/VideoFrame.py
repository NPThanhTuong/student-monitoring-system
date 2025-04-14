import base64
import json
import time
import uuid
import logging
import cv2
from kafka import KafkaProducer


class VideoFrame:
    """Class representing a video frame with its metadata"""

    def __init__(self, frame, camera_id, image_type="jpg"):
        self.frame = frame
        self.camera_id = camera_id
        self.image_type = image_type
        self.timestamp = int(time.time() * 1000)
        self.rows, self.cols = frame.shape[:2]
        self.base64_data = self._encode_frame()

    def _encode_frame(self):
        """Encode frame to base64 string"""
        _, buffer = cv2.imencode(f'.{self.image_type}', self.frame)
        return base64.b64encode(buffer).decode("utf-8")

    def to_dict(self):
        """Convert frame data to dictionary"""
        return {
            "cameraId": str(self.camera_id),
            "timestamp": self.timestamp,
            "rows": self.rows,
            "cols": self.cols,
            "type": self.image_type,
            "data": self.base64_data
        }


class VideoCapture:
    """Class for capturing video from a camera"""

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.camera_id = uuid.uuid4()
        self.cap = None

    def open(self):
        """Open the video capture device"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open webcam.")
        return self

    def read_frame(self):
        """Read a frame from the video capture device"""
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Camera is not open")

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")

        return VideoFrame(frame, self.camera_id)

    def close(self):
        """Close the video capture device"""
        if self.cap and self.cap.isOpened():
            self.cap.release()


class KafkaProducerWrapper:
    """Wrapper for Kafka producer"""

    def __init__(self, bootstrap_servers=None, topic=None):
        self.bootstrap_servers = bootstrap_servers or ['localhost:9092']
        self.topic = topic or 'video-stream-event'
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            max_block_ms=5000,
            value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )

    def send(self, message, topic=None):
        """Send message to Kafka topic"""
        target_topic = topic or self.topic
        self.producer.send(target_topic, value=message)
        self.producer.flush()


class VideoStreamer:
    """Main class for streaming video to Kafka"""

    def __init__(self, camera_index=0, kafka_servers=None, kafka_topic=None, delay=1):
        self.camera = VideoCapture(camera_index)
        self.kafka = KafkaProducerWrapper(kafka_servers, kafka_topic)
        self.delay = delay
        self.running = False

    def start(self):
        """Start streaming video frames to Kafka"""
        self.running = True
        self.camera.open()

        try:
            while self.running:
                try:
                    # Get frame and send to Kafka
                    frame = self.camera.read_frame()
                    self.kafka.send(frame.to_dict())
                    print(f"Sent data of camera: {self.camera.camera_id}")
                    time.sleep(self.delay)
                except Exception as e:
                    logging.error(f'An error occurred: {e}')
                    continue
        finally:
            self.stop()

    def stop(self):
        """Stop streaming"""
        self.running = False
        self.camera.close()
