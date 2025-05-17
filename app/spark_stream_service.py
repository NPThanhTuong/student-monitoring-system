import logging

from app.streaming_services.VideoStreamConsumer import VideoStreamConsumer


def stream_to_spark():
    """Main function to start the video stream consumer"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Create and start consumer
        consumer = VideoStreamConsumer()
        consumer.setup()
        consumer.start()
    except KeyboardInterrupt:
        logging.info("Application stopped by user")
    except Exception as e:
        logging.error(f"Error running application: {e}")

if __name__ == "__main__":
    stream_to_spark()
