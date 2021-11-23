from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

import json
import nltk
import numpy as np

import text_clean as clean

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split


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
		lemmatizeDataUDF = udf(clean.lemmatizeData)
		df = df.withColumn("lemmatized_text", lemmatizeDataUDF(df.text))
		#df.show()
		# collect lemmatized text & target as ndarray
		sentiment = np.array([int(row['target']) for row in df.collect()])
		text = np.array([str(row['lemmatized_text']) for row in df.collect()])
		
		X_text = vectorizer.transform(text)
		
		X_train, X_test, y_train, y_test = train_test_split(X_text, sentiment, test_size=0.15)
		
		
		
		classifier.partial_fit(X_train, y_train, classes=all_classes)
		
		
		print(classifier.score(X_test, y_test))
		
		
		
		
		
		#return rdd

		
#create sparkcontext and the streaming context
sc = SparkContext("local[2]", "recieveData")
ssc = StreamingContext(sc, 1)

#read the socket data 
lines = ssc.socketTextStream("localhost", 6100)

vectorizer = HashingVectorizer( decode_error='ignore', n_features=2**18, alternate_sign=False )
all_classes = np.array([0, 4])
classifier = SGDClassifier()


lines.foreachRDD(process)



ssc.start()
ssc.awaitTermination()
