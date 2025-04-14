import logging

from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, IntegerType


class KafkaStreamConsumer:
    """Handles Kafka connections and stream setup"""

    def __init__(self, spark_conn, bootstrap_servers="localhost:9092", topic="video-stream-event"):
        self.spark_conn = spark_conn
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.spark_df = None

    def connect_to_kafka(self):
        """Connect to Kafka and create a streaming DataFrame"""
        try:
            self.spark_df = self.spark_conn.readStream \
                .format('kafka') \
                .option('kafka.bootstrap.servers', self.bootstrap_servers) \
                .option('subscribe', self.topic) \
                .option('startingOffsets', 'earliest') \
                .load()
            logging.info("Kafka dataframe created successfully")
        except Exception as e:
            logging.warning(f"Kafka dataframe could not be created because: {e}")

        return self.spark_df

    def create_schema(self):
        """Create schema for parsing Kafka messages"""
        return StructType([
            StructField("cameraId", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("rows", IntegerType(), True),
            StructField("cols", IntegerType(), True),
            StructField("type", StringType(), True),
            StructField("data", StringType(), True),
        ])

    def create_selection_df(self):
        """Create a structured DataFrame from Kafka messages"""
        if self.spark_df is None:
            raise ValueError("Spark DataFrame not initialized. Call connect_to_kafka first.")

        schema = self.create_schema()
        selection_df = self.spark_df.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col('value'), schema).alias('data')).select("data.*")
        print(selection_df)

        return selection_df
