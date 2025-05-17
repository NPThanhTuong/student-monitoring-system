import datetime
import io
import logging

import cv2
from pyhdfs import HdfsClient

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

class HdfsHadoopSystem:
    def __init__(self):
        # HDFS Configuration
        self.hdfs_namenode_host = 'localhost'
        self.hdfs_namenode_port = '9870'
        self.hdfs_namenode_user = 'root'
        logger.info(f"Connecting to HDFS at {self.hdfs_namenode_host}:{self.hdfs_namenode_port}")
        self.hdfs_client = HdfsClient(hosts=f'{self.hdfs_namenode_host}:{self.hdfs_namenode_port}',
                                      user_name=self.hdfs_namenode_user,
                                      randomize_hosts=False)
        # Base directory for recognized faces in HDFS
        self.hdfs_base_dir = '/facial-recognition'
        self.ensure_hdfs_directory()

    def create_daily_hdfs_directory_path(self):
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        return f"{self.hdfs_base_dir}/{today}"

    def create_timestamp_file_path(self, prefix):
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        filename = f"{prefix}_{timestamp}.jpg"
        hdfs_path = f"{self.daily_dir}/{filename}"
        return hdfs_path

    def ensure_hdfs_directory(self):
        """Ensure the base directory exists in HDFS"""
        try:
            if not self.hdfs_client.exists(self.hdfs_base_dir):
                logger.info(f"Creating base directory {self.hdfs_base_dir} in HDFS")
                self.hdfs_client.mkdirs(self.hdfs_base_dir)

            # Create a directory for today's date
            self.daily_dir = self.create_daily_hdfs_directory_path()

            if not self.hdfs_client.exists(self.daily_dir):
                logger.info(f"Creating daily directory {self.daily_dir} in HDFS")
                self.hdfs_client.mkdirs(self.daily_dir)

        except Exception as e:
            logger.error(f"Error creating HDFS directories: {e}")
            raise

    def save_to_hdfs(self, frame, hdfs_path):
        """Save the recognized face frame to HDFS"""
        try:
            # Convert frame to a format suitable for HDFS storage
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                logger.error("Failed to encode image")
                return False

            # Generate filename with timestamp
            # hdfs_path = self.create_timestamp_file_path(person_name)

            # Convert buffer to byte array
            byte_data = io.BytesIO(buffer.tobytes())

            # Save to HDFS - using direct API call to avoid redirection issues
            logger.info(f"Saving image to HDFS: {hdfs_path}")
            self.hdfs_client.create(hdfs_path, byte_data.read(), overwrite=True)

            # Save a local copy as well
            # local_dir = os.environ.get('LOCAL_SAVE_DIR', '.\\saved_images')
            # if not os.path.exists(local_dir):
            #     os.makedirs(local_dir)
            #
            # filename = os.path.basename(hdfs_path)
            # local_path = os.path.join(local_dir, filename)
            # cv2.imwrite(local_path, frame)
            # logger.info(f"Saved local copy to: {local_path}")

            return True

        except Exception as e:
            logger.error(f"Error saving to HDFS: {e}")
            return False