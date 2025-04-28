import json
from datetime import datetime
import os

import cv2
import face_recognition
import numpy as np
import requests

from facial_recognition_services.FaceRecognitionDB import FaceRecognitionDB


class FaceRecognitionSystem:
    def __init__(self, db_config=None):
        self.reported_people = set()
        self.recognition_threshold = 0.6
        # self.last_results = None
        self.db = FaceRecognitionDB(db_config)
        # Recognition parameters
        self.face_detection_model = "hog"  # options: "hog" (faster) or "cnn" (more accurate)
        self.recognition_tolerance = 0.6  # lower = more strict

    def add_face_from_image(self, name, image_path):
        """Add a face from an image file"""
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)

            # Detect faces
            face_locations = face_recognition.face_locations(image, model=self.face_detection_model)

            if not face_locations:
                print(f"No faces found in {image_path}")
                return False

            # Get face encodings
            face_encodings = face_recognition.face_encodings(image, face_locations)

            if not face_encodings:
                print(f"Could not encode face in {image_path}")
                return False

            # Add person to database
            person_id = self.db.add_person(name)
            if person_id is None:
                return False

            # Add face encoding to database
            encoding_id = self.db.add_face_encoding(person_id, face_encodings[0])

            return encoding_id is not None
        except Exception as e:
            print(f"Error adding face from image: {e}")
            return False

    def add_face_from_webcam(self, name):
        """Capture and add a face using webcam"""
        try:
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                print("Error: Could not open webcam")
                return False

            print("Position your face in the frame and press SPACE to capture or ESC to cancel")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Display helper box in the center
                height, width, _ = frame.shape
                center_x, center_y = width // 2, height // 2
                box_size = min(width, height) // 2

                # Draw guide box
                cv2.rectangle(frame,
                              (center_x - box_size // 2, center_y - box_size // 2),
                              (center_x + box_size // 2, center_y + box_size // 2),
                              (0, 255, 0), 2)

                # Draw instructions
                cv2.putText(frame, "Position face inside box", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "SPACE: Capture, ESC: Cancel", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Show frame
                cv2.imshow("Capture Face", frame)

                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    cap.release()
                    cv2.destroyAllWindows()
                    return False
                elif key == 32:  # SPACE
                    # Convert frame from BGR to RGB (for face_recognition)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Detect faces
                    face_locations = face_recognition.face_locations(rgb_frame, model=self.face_detection_model)

                    if not face_locations:
                        print("No face detected. Please try again.")
                        continue

                    # Get face encodings
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                    if not face_encodings:
                        print("Could not encode face. Please try again.")
                        continue

                    # Add person to database
                    person_id = self.db.add_person(name)
                    if person_id is None:
                        cap.release()
                        cv2.destroyAllWindows()
                        return False

                    # Add face encoding to database
                    encoding_id = self.db.add_face_encoding(person_id, face_encodings[0])

                    # Save face image for reference (optional)
                    save_dir = "face_images"
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = os.path.join(save_dir, f"{name}_{timestamp}.jpg")
                    cv2.imwrite(image_path, frame)
                    print(f"Face image saved to {image_path}")

                    cap.release()
                    cv2.destroyAllWindows()
                    return encoding_id is not None

            cap.release()
            cv2.destroyAllWindows()
            return False
        except Exception as e:
            print(f"Error capturing face: {e}")
            cv2.destroyAllWindows()
            return False

    def delete_person(self, name):
        """Delete a person from the database"""
        return self.db.delete_person(name)

    def list_all_persons(self):
        """List all persons in the database"""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT p.name, COUNT(f.id) as face_count 
                FROM persons p
                LEFT JOIN face_encodings f ON p.id = f.person_id
                GROUP BY p.name
                ORDER BY p.name
            """)

            results = cursor.fetchall()
            cursor.close()
            conn.close()

            if not results:
                print("No persons found in the database")
                return []

            print("\nPersons in database:")
            print("--------------------")
            for name, face_count in results:
                print(f"{name}: {face_count} face encoding(s)")

            return [name for name, _ in results]
        except Exception as e:
            print(f"Error listing persons: {e}")
            return []

    def recognize_from_webcam(self):
        """Real-time face recognition from webcam using vector similarity"""
        # Check if we have faces to compare against
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM face_encodings")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()

        if count == 0:
            print("No face encodings found in database. Please add faces first.")
            return

        # Load all encodings for traditional comparison method (backup)
        known_names, known_encodings = self.db.get_all_face_encodings()

        # Start webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        print("Face Recognition Started")
        print("Press 'q' to quit")

        # For performance, only process every other frame
        process_this_frame = True

        # For tracking recognized people
        recognized_names = set()
        recognition_counts = {}  # Track how many times each person is recognized

        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                break

            # Process only every other frame for better performance
            if process_this_frame:
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert from BGR to RGB
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Find all faces in the current frame
                face_locations = face_recognition.face_locations(rgb_small_frame, model=self.face_detection_model)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                face_confidences = []

                for face_encoding in face_encodings:
                    try:
                        # Try to use vector similarity search (faster and more accurate)
                        similar_faces = self.db.find_similar_faces(face_encoding, self.recognition_tolerance)

                        if similar_faces:
                            # Get the most similar face
                            name, confidence = similar_faces[0]

                            # Track recognition
                            recognized_names.add(name)
                            recognition_counts[name] = recognition_counts.get(name, 0) + 1
                        else:
                            name = "Unknown"
                            confidence = 0

                        face_names.append(name)
                        face_confidences.append(confidence)

                    except Exception as e:
                        print(f"Error with vector search, falling back to traditional method: {e}")

                        # Fallback to traditional method
                        matches = face_recognition.compare_faces(known_encodings, face_encoding,
                                                                 tolerance=self.recognition_tolerance)
                        name = "Unknown"
                        confidence = 0

                        if True in matches:
                            # Find best match
                            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances)

                            if matches[best_match_index]:
                                name = known_names[best_match_index]
                                confidence = 1 - face_distances[best_match_index]

                                # Track recognition
                                recognized_names.add(name)
                                recognition_counts[name] = recognition_counts.get(name, 0) + 1

                        face_names.append(name)
                        face_confidences.append(confidence)

            process_this_frame = not process_this_frame

            # Display results
            for (top, right, bottom, left), name, confidence in zip(face_locations, face_names, face_confidences):
                # Scale back up face locations
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Set color based on confidence (green for known, red for unknown)
                if name != "Unknown":
                    # Gradient from yellow to green based on confidence
                    green = int(255 * min(confidence * 1.5, 1.0))
                    red = int(255 * max(1 - (confidence - 0.5) * 2, 0) if confidence > 0.5 else 255)
                    color = (0, green, red)  # OpenCV uses BGR

                    # Format confidence as percentage
                    confidence_text = f"{confidence * 100:.1f}%"
                else:
                    color = (0, 0, 255)  # Red for unknown
                    confidence_text = ""

                # Draw box around face
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                # Draw label with name
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

                # Draw confidence if known
                if name != "Unknown":
                    cv2.putText(frame, confidence_text, (left + 6, top - 6), font, 0.5, color, 1)

            # Add info text
            cv2.putText(frame, "Press 'q' to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display the frame
            cv2.imshow('Face Recognition', frame)

            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Clean up
        cap.release()
        cv2.destroyAllWindows()

        # Print summary
        if recognized_names:
            print("\nRecognized persons:")
            print("-----------------")
            for name in recognized_names:
                count = recognition_counts.get(name, 0)
                print(f"{name}: detected {count} times")
        else:
            print("No persons were recognized")

    def recognize_from_spark_streaming(self, frame, known_names, known_encodings):
        """Real-time face recognition from Spark streaming using vector similarity"""
        try:

            # Process face recognition
            recognition_results = self._process_face_recognition(frame, known_names, known_encodings)

            # Send recognized faces to API
            self._report_recognized_faces(recognition_results)

            # Display the frame (optional - can be removed if no visual feedback needed)
            cv2.imshow('Face Recognition', frame)
            cv2.waitKey(1)
        except Exception as e:
            print(f"Error occurs while recognize from spark streaming: {e}")

    def _process_face_recognition(self, frame, known_names, known_encodings):
        """Process face recognition and return detected faces with their information"""
        # Initialize class variables if not already set
        if not hasattr(self, '_process_this_frame'):
            self._process_this_frame = True

        # Process only every other frame for better performance
        results = []

        if self._process_this_frame:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert from BGR to RGB
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find all faces in the current frame
            face_locations = face_recognition.face_locations(rgb_small_frame,
                                                             model=self.face_detection_model)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for i, face_encoding in enumerate(face_encodings):
                name, confidence = self._identify_face(face_encoding, known_names, known_encodings)

                # Scale back up face location
                top, right, bottom, left = face_locations[i]
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Add to results
                results.append({
                    'location': (top, right, bottom, left),
                    'name': name,
                    'confidence': confidence,
                    'timestamp': self._get_current_timestamp()
                })

        # Toggle the processing flag
        self._process_this_frame = not self._process_this_frame

        return results

    def _identify_face(self, face_encoding, known_names, known_encodings):
        """Identify a face using vector similarity or traditional method as fallback"""
        try:
            # Try to use vector similarity search (faster and more accurate)
            similar_faces = self.db.find_similar_faces(face_encoding,
                                                       self.recognition_tolerance)

            if similar_faces:
                # Get the most similar face
                name, confidence = similar_faces[0]
            else:
                name = "Unknown"
                confidence = 0

        except Exception as e:
            print(f"Error with vector search, falling back to traditional method: {e}")

            # Fallback to traditional method
            matches = face_recognition.compare_faces(known_encodings, face_encoding,
                                                     tolerance=self.recognition_tolerance)
            name = "Unknown"
            confidence = 0

            if True in matches:
                # Find best match
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]

        return name, confidence

    def _get_current_timestamp(self):
        """Get current timestamp in the desired format"""
        return datetime.now().isoformat()

    def _report_recognized_faces(self, recognition_results):
        """Report recognized faces to the API, ensuring each person is only reported once"""
        # Initialize set to track reported people if not already created
        if not hasattr(self, 'reported_people'):
            self.reported_people = set()

        # Set default threshold if not already defined
        if not hasattr(self, 'recognition_threshold'):
            self.recognition_threshold = 0.6

        API_ENDPOINT = "http://127.0.0.1:9191/post/observations"
        DATASTREAM_ID = 2
        API_HEADERS = {'Content-Type': 'application/json'}
        API_BASIC_AUTH = ('sensor', 'sensor')

        for result in recognition_results:
            name = result.get('name')
            confidence = result.get('confidence', 0)

            # Only report known faces that haven't been reported yet
            if (name and name != "Unknown" and
                    name not in self.reported_people and
                    confidence >= self.recognition_threshold):
                try:
                    # Prepare data for API
                    recognition_data = {
                        "Datastream": {
                            "id": DATASTREAM_ID
                        },
                        "result": [
                            {
                                "name": name,
                                "confidence": confidence,
                                "timestamp": result.get('timestamp', datetime.now().isoformat())
                            }
                        ]
                    }

                    # Send POST request to API
                    response = requests.post(API_ENDPOINT, headers=API_HEADERS, json=recognition_data, auth=API_BASIC_AUTH)
                    response.raise_for_status()

                    # Check if request was successful
                    print(f"Successfully reported recognition of {name}")
                    # Add to set of reported people to avoid duplicates
                    self.reported_people.add(name)
                except requests.exceptions.RequestException as e:
                    print(f"API request failed: {e}")
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON response: {e}")
                except Exception as e:
                    print(f"Error sending recognition data to API: {e}")
