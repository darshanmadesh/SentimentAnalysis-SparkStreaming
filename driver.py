from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
import json
import nltk


def getSparkSessionInstance(sparkConf):
	if ("sparkSessionSingletonInstance" not in globals()):
		globals()["sparkSessionSingletonInstance"] = SparkSession\
			.builder\
			.config(conf=sparkConf)\
			.getOrCreate()
	return globals()["sparkSessionSingletonInstance"]

sc = SparkContext("local[2]", "recieveData")
# Create a streaming context
ssc = StreamingContext(sc, 1)

lines = ssc.socketTextStream("localhost", 6100)



def readData(rdd):
	if not rdd.isEmpty():
		spark = getSparkSessionInstance(rdd.context.getConf())
	    # json to dict
		rdd = rdd.map(lambda x : json.loads(x))
		# get only the instance as list
		rdd = rdd.flatMap(lambda d: list(d[k] for k in d))
		rdd = rdd.map(lambda d: list(d[k] for k in d))
		#df = rdd.map(lambda x : (x[0], x[1])).toDF(("target", "feature"))
		return rdd
    


lines.foreachRDD(readData)

ssc.start()
ssc.awaitTermination()
