from pyspark import SparkContext
from pyspark.streaming import StreamingContext
import json


sc = SparkContext("local[*]", "recieveData")
# Create a streaming context
ssc = StreamingContext(sc, 1)

lines = ssc.socketTextStream("localhost", 6100)



def todf(rdd):
    rdd = rdd.map(lambda x : json.loads(x))
    data = rdd.collect()
    for row in data:
        print(row)

def readMyStream(rdd):
  if not rdd.isEmpty():
    df = spark.read.json(rdd)
    print('Started the Process')
    print('Selection of Columns')
    df = df.select('t1','t2','t3','timestamp').where(col("timestamp").isNotNull())
    df.show()


lines.foreachRDD(todf)

ssc.start()
ssc.awaitTermination()
