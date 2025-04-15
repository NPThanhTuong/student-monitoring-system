import base64
import logging

import cv2
import face_recognition
import numpy as np

from facial_recognition_services.FaceRecognitionDB import FaceRecognitionDB
from facial_recognition_services.FaceRecognitionSystem import FaceRecognitionSystem


class ImageProcessor:
    """Process images from base64 data and perform face recognition"""
    def __init__(self, window_name="Face Recognition"):
        self.window_name = window_name

    @staticmethod
    def process_row(face_system, row, known_names, known_encodings):
        """Process a single row of data containing a base64 encoded image"""
        try:
            # Decode base64 to image
            base64_data = row["data"]
            img_data = base64.b64decode(base64_data)
            np_arr = np.frombuffer(img_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
                face_system.recognize_from_spark_streaming(frame, known_names, known_encodings)
        except Exception as e:
            logging.error(f"Error processing frame: {e}")

    @staticmethod
    def create_batch_processor(known_names, known_encodings, face_system):
        def process_batch(batch_df, batch_id):
            """Process a batch of data (called by Spark streaming)"""
            rows = batch_df.select("data").collect()

            for row in rows:
                ImageProcessor.process_row(face_system, row, known_names, known_encodings)

        return process_batch
