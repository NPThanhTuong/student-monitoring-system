import logging

from pyspark.sql import SparkSession


class SparkConnectionManager:
    """Manages Spark connection for the application"""

    def __init__(self, app_name="SparkDataStreaming"):
        self.app_name = app_name
        self.spark_conn = None

    def create_connection(self):
        """Create and return a Spark session"""
        try:
            self.spark_conn = SparkSession.builder \
                .appName(self.app_name) \
                .config('spark.jars.packages', "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1") \
                .getOrCreate()

            self.spark_conn.sparkContext.setLogLevel("ERROR")
            logging.info("Spark connection created successfully!")
        except Exception as e:
            logging.error(f"Couldn't create the spark session due to exception {e}")

        return self.spark_conn
