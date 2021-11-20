from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

import json
import nltk
import numpy as np

import text_clean as clean


def getSparkSessionInstance(sparkConf):
	if ("sparkSessionSingletonInstance" not in globals()):
		globals()["sparkSessionSingletonInstance"] = SparkSession\
			.builder\
			.config(conf=sparkConf)\
			.getOrCreate()
	return globals()["sparkSessionSingletonInstance"]


def process(rdd):
	if not rdd.isEmpty():
	
		spark = getSparkSessionInstance(rdd.context.getConf())
		
	    # json to dict
		rdd = rdd.map(lambda x : json.loads(x))
		
		# get only the instance as list
		rdd = rdd.flatMap(lambda d: list(d[k] for k in d))
		rdd = rdd.map(lambda d: list(d[k] for k in d))
		
		#rdd = rdd.map(lambda x : x + (clean.cleanData(str(x[1]))))
		
		#data cleaning
		# convert rdd to dataframe and get the cleaned data.
		df = rdd.map(lambda x : (x[0], x[1])).toDF(("target", "text"))
		cleanDataUDF = udf(clean.cleanData)
		df = df.withColumn("clean_text", cleanDataUDF(df.text))
		
		# collect clean text & target as ndarray
		sentiment = np.array([int(row['target']) for row in df.collect()])
		text = np.array([str(row['clean_text']) for row in df.collect()])
		
		
		
		return rdd

		
#create sparkcontext and the streaming context
sc = SparkContext("local[2]", "recieveData")
ssc = StreamingContext(sc, 1)

#read the socket data 
lines = ssc.socketTextStream("localhost", 6100)



lines.foreachRDD(process)



ssc.start()
ssc.awaitTermination()
