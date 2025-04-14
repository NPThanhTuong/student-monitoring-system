import base64
import logging

import cv2
import face_recognition
import numpy as np


class ImageProcessor:
    """Process images from base64 data and perform face recognition"""

    def __init__(self, window_name="Face Recognition"):
        self.window_name = window_name

    def process_row(self, row):
        """Process a single row of data containing a base64 encoded image"""
        try:
            # Decode base64 to image
            base64_data = row["data"]
            img_data = base64.b64decode(base64_data)
            np_arr = np.frombuffer(img_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
                # Convert from BGR (OpenCV) to RGB (required by face_recognition)
                rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect faces
                face_locations = face_recognition.face_locations(rgb_img)

                # Draw rectangles around faces
                for top, right, bottom, left in face_locations:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Display image with detected faces
                cv2.imshow(self.window_name, frame)
                cv2.waitKey(1)
        except Exception as e:
            logging.error(f"Error processing frame: {e}")

    @staticmethod
    def process_batch(batch_df, batch_id):
        """Process a batch of data (called by Spark streaming)"""
        processor = ImageProcessor()
        rows = batch_df.select("data").collect()
        for row in rows:
            processor.process_row(row)
