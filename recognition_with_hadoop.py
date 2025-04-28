#!/usr/bin/env python3
"""
Facial Recognition Application with Hadoop HDFS Integration (Windows Host Version)
"""

import os
import cv2
import face_recognition
import numpy as np
import time
import datetime
from pyhdfs import HdfsClient
import logging
import io
from PIL import Image
import socket
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("facial_recognition.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FacialRecognitionSystem:
    def __init__(self):
        # HDFS Configuration
        self.hdfs_namenode_host = os.environ.get('HDFS_NAMENODE_HOST', 'localhost')
        self.hdfs_namenode_port = os.environ.get('HDFS_NAMENODE_PORT', '9870')
        logger.info(f"Connecting to HDFS at {self.hdfs_namenode_host}:{self.hdfs_namenode_port}")
        self.hdfs_client = HdfsClient(hosts=f'{self.hdfs_namenode_host}:{self.hdfs_namenode_port}',
                                      user_name='root',
                                      randomize_hosts = False)

        # Base directory for recognized faces in HDFS
        self.hdfs_base_dir = '/facial-recognition'
        self.ensure_hdfs_directory()

        # Load known faces and their encodings
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

        # Camera configuration
        self.camera_id = int(os.environ.get('CAMERA_ID', 0))
        self.frame_rate = int(os.environ.get('FRAME_RATE', 30))  # Frames per second

        # Face recognition parameters
        self.face_recognition_model = os.environ.get('FACE_MODEL',
                                                     "hog")  # or "cnn" for better accuracy but slower
        self.tolerance = float(os.environ.get('FACE_TOLERANCE', 0.6))

    def ensure_hdfs_directory(self):
        """Ensure the base directory exists in HDFS"""
        try:
            if not self.hdfs_client.exists(self.hdfs_base_dir):
                logger.info(f"Creating base directory {self.hdfs_base_dir} in HDFS")
                self.hdfs_client.mkdirs(self.hdfs_base_dir)

            # Create a directory for today's date
            today = datetime.datetime.now().strftime('%Y-%m-%d')
            self.daily_dir = f"{self.hdfs_base_dir}/{today}"

            if not self.hdfs_client.exists(self.daily_dir):
                logger.info(f"Creating daily directory {self.daily_dir} in HDFS")
                self.hdfs_client.mkdirs(self.daily_dir)

        except Exception as e:
            logger.error(f"Error creating HDFS directories: {e}")
            raise

    def load_known_faces(self):
        """Load known faces from the 'known_faces' directory"""
        known_faces_dir = os.environ.get('KNOWN_FACES_DIR', ".\\data\\known_faces")

        if not os.path.exists(known_faces_dir):
            logger.warning(f"Known faces directory {known_faces_dir} does not exist. Creating it.")
            os.makedirs(known_faces_dir)
            logger.info(f"Please add face images to {known_faces_dir} directory.")
            return

        logger.info("Loading known faces...")
        for filename in os.listdir(known_faces_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                face_image_path = os.path.join(known_faces_dir, filename)
                try:
                    # Get person name from filename (without extension)
                    person_name = os.path.splitext(filename)[0]

                    # Load and encode face
                    face_image = face_recognition.load_image_file(face_image_path)
                    face_encodings = face_recognition.face_encodings(face_image)

                    if len(face_encodings) > 0:
                        face_encoding = face_encodings[0]
                        self.known_face_encodings.append(face_encoding)
                        self.known_face_names.append(person_name)
                        logger.info(f"Loaded face: {person_name}")
                    else:
                        logger.warning(f"No face found in {filename}")

                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")

        logger.info(f"Loaded {len(self.known_face_encodings)} known faces")

    def save_to_hdfs(self, frame, person_name):
        """Save the recognized face frame to HDFS"""
        try:
            # Convert frame to a format suitable for HDFS storage
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                logger.error("Failed to encode image")
                return False

            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            filename = f"{person_name}_{timestamp}.jpg"
            hdfs_path = f"{self.daily_dir}/{filename}"

            # Convert buffer to byte array
            byte_data = io.BytesIO(buffer.tobytes())

            # Save to HDFS - using direct API call to avoid redirection issues
            logger.info(f"Saving image to HDFS: {hdfs_path}")
            self.hdfs_client.create(hdfs_path, byte_data.read(), overwrite=True)

            # Save a local copy as well
            local_dir = os.environ.get('LOCAL_SAVE_DIR', '.\\saved_images')
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)

            local_path = os.path.join(local_dir, filename)
            cv2.imwrite(local_path, frame)
            logger.info(f"Saved local copy to: {local_path}")

            return True

        except Exception as e:
            logger.error(f"Error saving to HDFS: {e}")
            return False

    def process_frame(self, frame):
        """Process a single frame and identify faces"""
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert from BGR (OpenCV format) to RGB (face_recognition format)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_small_frame,
                                                         model=self.face_recognition_model)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        recognized_person = None

        for face_encoding in face_encodings:
            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, tolerance=self.tolerance
            )
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings,
                                                            face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    recognized_person = name

            face_names.append(name)

        # Draw boxes and labels on the frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back face locations (they were scaled down by 1/4)
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw face box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw label background
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)

            # Add text
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 0), 1)

        return frame, recognized_person

    def run(self):
        """Run the facial recognition system"""
        if len(self.known_face_encodings) == 0:
            logger.warning(
                "No known faces loaded. Please add face images to the known_faces directory.")
            return

        logger.info("Starting webcam...")
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)  # Use DirectShow on Windows

        if not cap.isOpened():
            logger.error(f"Failed to open webcam {self.camera_id}")
            return

        logger.info("Starting facial recognition...")
        successful_recognition = False

        try:
            while not successful_recognition:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to grab frame")
                    break

                # Process frame
                frame, recognized_person = self.process_frame(frame)

                # Display the frame
                cv2.imshow('Facial Recognition', frame)

                # If person is recognized, save to HDFS and exit
                if recognized_person and recognized_person != "Unknown":
                    logger.info(f"Person recognized: {recognized_person}")

                    # Save the frame to HDFS
                    if self.save_to_hdfs(frame, recognized_person):
                        logger.info("Successfully saved to HDFS")
                        successful_recognition = True
                    else:
                        logger.error("Failed to save to HDFS")

                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Control frame rate
                time.sleep(1 / self.frame_rate)

        finally:
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Facial recognition system shutdown")


def check_docker_connection():
    """Check if we can connect to Docker containers"""
    namenode_host = os.environ.get('HDFS_NAMENODE_HOST', 'localhost')
    namenode_port = int(os.environ.get('HDFS_NAMENODE_PORT', 9000))

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((namenode_host, namenode_port))
        sock.close()

        if result == 0:
            logger.info(
                f"Successfully connected to HDFS namenode at {namenode_host}:{namenode_port}")
            return True
        else:
            logger.error(f"Could not connect to HDFS namenode at {namenode_host}:{namenode_port}")
            return False
    except Exception as e:
        logger.error(f"Error checking connection to HDFS: {e}")
        return False


if __name__ == "__main__":
    try:
        # Check Docker connectivity
        if not check_docker_connection():
            logger.error(
                "Cannot connect to HDFS namenode. Make sure Docker containers are running.")
            logger.info("Run 'docker-compose up -d' to start the HDFS containers.")
            input("Press Enter to exit...")
            exit(1)

        # Run the facial recognition system
        facial_recognition_system = FacialRecognitionSystem()
        facial_recognition_system.run()
    except Exception as e:
        logger.error(f"Error running facial recognition system: {e}")
        import traceback

        traceback.print_exc()
        input("Press Enter to exit...")