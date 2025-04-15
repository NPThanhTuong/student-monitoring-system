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
    def process_row(row, known_names, known_encodings):
        """Process a single row of data containing a base64 encoded image"""
        try:
            # Decode base64 to image
            base64_data = row["data"]
            img_data = base64.b64decode(base64_data)
            np_arr = np.frombuffer(img_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            face_system = FaceRecognitionSystem()

            if frame is not None:
                face_system.recognize_from_spark_streaming(frame, known_names, known_encodings)
        except Exception as e:
            logging.error(f"Error processing frame: {e}")

    @staticmethod
    def process_batch(batch_df, batch_id):
        """Process a batch of data (called by Spark streaming)"""
        processor = ImageProcessor()
        rows = batch_df.select("data").collect()

        # Check if we have faces to compare against
        db = FaceRecognitionDB()
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM face_encodings")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()

        if count == 0:
            print("No face encodings found in database. Please add faces first.")
            return

        known_names, known_encodings = db.get_all_face_encodings()

        for row in rows:
            processor.process_row(row, known_names, known_encodings)
