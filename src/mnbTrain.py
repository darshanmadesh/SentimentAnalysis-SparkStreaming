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
from sklearn.naive_bayes import MultinomialNB
import joblib


def getSparkSessionInstance(sparkConf):
	if ("sparkSessionSingletonInstance" not in globals()):
		globals()["sparkSessionSingletonInstance"] = SparkSession\
			.builder\
			.config(conf=sparkConf)\
			.getOrCreate()
	return globals()["sparkSessionSingletonInstance"]


def process(time, rdd):
	if not rdd.isEmpty():
		print(f"======== {time} ========")
		
		spark = getSparkSessionInstance(rdd.context.getConf())
		
	    	# json to dict
		rdd = rdd.map(lambda x : json.loads(x))
		
		# get only the instance as list
		rdd = rdd.flatMap(lambda d: list(d[k] for k in d))
		rdd = rdd.map(lambda d: list(d[k] for k in d))
		rdd = rdd.map(lambda i:[1,i[1]] if i[0] == 4 else [0,i[1]])
		#data cleaning & preprocessing -
		
		# convert rdd to dataframe and get the cleaned & lemmatized data.
		df = rdd.map(lambda x : (x[0], x[1])).toDF(("target", "text"))
		lemmatizeDataUDF = udf(clean.lemmatizeData)
		df = df.withColumn("lemmatized_text", lemmatizeDataUDF(df.text))
		
		# collect lemmatized text & target as ndarray
		sentiment = np.array([int(row['target']) for row in df.collect()])
		text = np.array([str(row['lemmatized_text']) for row in df.collect()])
		
		print(len(text))
		
		# vectorize the text using hashing vectorizer
		X_text = vectorizer.transform(text)
		
		# partially fit the batch
		for cls_name, cls in mnb_classifiers.items():
			cls.partial_fit(X_text, sentiment, classes=np.unique(sentiment))
			
		
		
		
		

if __name__ == '__main__':
	#create sparkcontext and the streaming context
	sc = SparkContext("local[2]", "MNBClassifier")
	ssc = StreamingContext(sc, 1)

	#read the socket data 
	lines = ssc.socketTextStream("localhost", 6100)
	
	# Create the vectorizer
	vectorizer = HashingVectorizer( decode_error='ignore', n_features=2**20, alternate_sign=False )
	
	all_classes = np.array([0, 4])
	
	# create Multinimial Naive Bayes Classifiers
	mnb_classifiers = {
		"MNB1" : MultinomialNB( alpha = 1.0, fit_prior = True, class_prior = None ),
		"MNB2" : MultinomialNB( alpha = 2.0, fit_prior = True, class_prior = None ),
		"MNB3" : MultinomialNB( alpha = 3.0, fit_prior = True, class_prior = None ),
		"MNB4" : MultinomialNB( alpha = 4.0, fit_prior = True, class_prior = None )
	}


	lines.foreachRDD(process)
	

	ssc.start()
	
	
	# wait for streaming to finish
	ssc.awaitTermination(timeout=60*20)
	
	ssc.stop(stopGraceFully=True)
	
	
	for cls_name, cls in mnb_classifiers.items():
		joblib.dump(cls, f"{cls_name}.pkl")
		print(f"{cls_name} saved successfully!!!")
		
	
	
	
