from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split

spark = SparkSession.builder.appName("StructuredStreaming").getOrCreate()

records = spark.readStream \
        .format("socket") \
        .option("host", "localhost") \
        .option("port", 6100) \
        .load()
        
records.

