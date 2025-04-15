from facial_recognition_services.FaceRecognitionDB import FaceRecognitionDB
from facial_recognition_services.FaceRecognitionSystem import FaceRecognitionSystem
from streaming_services.ImageProcessor import ImageProcessor
from streaming_services.KafkaStreamConsumer import KafkaStreamConsumer
from streaming_services.SparkConnectionManager import SparkConnectionManager


class VideoStreamConsumer:
    """Main class that coordinates the video stream consumption pipeline"""

    def __init__(self, app_name="VideoStreamConsumer", bootstrap_servers="localhost:9092", topic="video-stream-event"):
        self.app_name = app_name
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.spark_manager = SparkConnectionManager(app_name)
        self.spark_conn = None
        self.kafka_consumer = None

    def setup(self):
        """Set up all connections and dataframes"""
        # Create Spark connection
        self.spark_conn = self.spark_manager.create_connection()
        if self.spark_conn is None:
            raise RuntimeError("Failed to create Spark connection")

        # Create Kafka consumer
        self.kafka_consumer = KafkaStreamConsumer(
            self.spark_conn,
            bootstrap_servers=self.bootstrap_servers,
            topic=self.topic
        )

    def start(self):
        """Start consuming and processing video stream"""
        # Connect to Kafka
        spark_df = self.kafka_consumer.connect_to_kafka()
        if spark_df is None:
            raise RuntimeError("Failed to connect to Kafka")

        # Create selection dataframe
        selection_df = self.kafka_consumer.create_selection_df()


        db = FaceRecognitionDB()
        face_system = FaceRecognitionSystem()

        # Check if we have faces to compare against
        if db.is_face_encoding_storage_empty():
            print("No face encodings found in database. Please add faces first.")
            return

        known_names, known_encodings = db.get_all_face_encodings()

        # To pass more parameter to process_batch func (through closure function)
        batch_processor = ImageProcessor.create_batch_processor(known_names, known_encodings, face_system)

        # Start stream processing
        query = selection_df.writeStream \
            .foreachBatch(batch_processor) \
            .start()

        # Wait for query termination
        query.awaitTermination()