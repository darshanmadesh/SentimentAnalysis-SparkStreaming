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
from sklearn.linear_model import Perceptron
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
		for cls_name, cls in perc_classifiers.items():
			cls.partial_fit(X_text, sentiment, classes=np.unique(sentiment))
			
		
		
		
		

if __name__ == '__main__':
	#create sparkcontext and the streaming context
	sc = SparkContext("local[2]", "SGDClassifier")
	ssc = StreamingContext(sc, 1)

	#read the socket data 
	lines = ssc.socketTextStream("localhost", 6100)
	
	# Create the vectorizer
	vectorizer = HashingVectorizer( decode_error='ignore', n_features=2**20, alternate_sign=False )
	
	all_classes = np.array([0, 4])
	
	# create Perceptron Classifiers
	perc_classifiers = {
		"PERC1" : Perceptron(),
		"PERC2" : Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, shuffle=False, verbose=0, eta0=1.0, n_jobs=1, random_state=0, class_weight=None),
		"PERC3" : Perceptron(penalty=None, alpha=0.002, fit_intercept=True, shuffle=True, verbose=0, eta0=1.0, n_jobs=1, random_state=0, class_weight=None),
		"PERC4" : Perceptron(penalty=None, alpha=0.0076, fit_intercept=True, shuffle=False, verbose=0, eta0=1.0, n_jobs=1, random_state=0, class_weight=None),
		"PERC5" : Perceptron(penalty=None, alpha=0.02, fit_intercept=True, shuffle=False, verbose=0, eta0=1.0, n_jobs=1, random_state=0, class_weight=None)
	}


	lines.foreachRDD(process)
	

	ssc.start()
	
	
	# wait for streaming to finish
	ssc.awaitTermination(timeout=60*25)
	
	ssc.stop(stopGraceFully=True)
	
	
	for cls_name, cls in perc_classifiers.items():
		joblib.dump(cls, f"{cls_name}.pkl")
		print(f"{cls_name} saved successfully!!!")
		
	
	
	
