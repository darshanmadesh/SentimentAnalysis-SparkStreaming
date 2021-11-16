from pyspark import SparkContext
from pyspark.streaming import StreamingContext
import json


sc = SparkContext("local[2]", "recieveData")
# Create a streaming context
ssc = StreamingContext(sc, 1)

lines = ssc.socketTextStream("localhost", 6100)



def todf(rdd):
	# json to dict
    rdd = rdd.map(lambda x : json.loads(x))
    # get only the instance as list
    rdd = rdd.flatMap(lambda d: list(d[k] for k in d))
    rdd = rdd.map(lambda d: list(d[k] for k in d))
    


lines.foreachRDD(todf)

ssc.start()
ssc.awaitTermination()
