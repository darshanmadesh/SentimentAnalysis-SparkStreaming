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
from sklearn.linear_model import PassiveAggressiveClassifier
import joblib

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import matplotlib.pyplot as plt



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
		rdd = rdd.map(lambda i:[1, i[1]] if i[0] == 4 else [0, i[1]])
		
		# Data cleaning and preprocessing -
		#data cleaning
		# convert rdd to dataframe and get the cleaned data.
		df = rdd.map(lambda x : (x[0], x[1])).toDF(("target", "text"))
		lemmatizeDataUDF = udf(clean.lemmatizeData)
		df = df.withColumn("lemmatized_text", lemmatizeDataUDF(df.text))
		#df.show()

		# collect lemmatized text & target as ndarray
		sentiment = np.array([int(row['target']) for row in df.collect()])
		text = np.array([str(row['lemmatized_text']) for row in df.collect()])
		
		print(f"batch size : {len(text)}")
		#print(f"type : {text.dtype}")
		
		# vectorize cleaned text using hashing vectorizer
		X_test = vectorizer.transform(text)
		print(f"type : {X_test.dtype}")
		print(f"shape : {X_test.shape}")
		print(f"sentiment type : {sentiment.dtype}")
		print(f"sentiment shape : {sentiment.shape}")
		
		
		
		
		# predict and get various performance metrics
		for cls_name, cls in pa_models.items():
			pred = cls["model"].predict(X_test)
			cls["scores"]["accuracy"].append(accuracy_score(sentiment, pred))
			cls["scores"]["precision"].append(precision_score(sentiment, pred))
			cls["scores"]["recall"].append(recall_score(sentiment, pred))
			cls["scores"]["f1"].append(f1_score(sentiment, pred))
			#print(f"{cls_name} : {cls['scores']}")
		
		
		

if __name__ == '__main__':
	#create sparkcontext and the streaming context
	sc = SparkContext("local[2]", "recieveData")
	ssc = StreamingContext(sc, 1)

	#read the socket data 
	lines = ssc.socketTextStream("localhost", 6100)

	vectorizer = HashingVectorizer( decode_error='ignore', n_features=2**20, alternate_sign=False )
	
	pa_models = {
		"PA1" : {
			"model" : None,
			"scores" : {
				"accuracy" : [],
				"precision" : [],
				"recall" : [],
				"f1" :  []
			}
		},
		"PA2" : {
			"model" : None,
			"scores" : {
				"accuracy" : [],
				"precision" : [],
				"recall" : [],
				"f1" :  []
			}
		},
		"PA3" : {
			"model" : None,
			"scores" : {
				"accuracy" : [],
				"precision" : [],
				"recall" : [],
				"f1" :  []
			}
		}
	}
	
	for cls_name in pa_models :
		pa_models[cls_name]["model"] = joblib.load(f"{cls_name}.pkl")
	

	lines.foreachRDD(process)



	ssc.start()
	ssc.awaitTermination(timeout=60*20)
	
	ssc.stop(stopGraceFully=True)

	with open('score.txt', 'w+') as f:
		f.write(str(pa_models))
	



	
	def gen_lists(pa_models):
	    x_acc, x_pre, x_rec, x_f1 = {}, {}, {}, {}
	    for k,v in pa_models.items():
	        for k2, v2 in v.items():
	            if (k2 == 'scores'):
	                for k3, v3 in v2.items():
	                    x_acc[k] = v2['accuracy']
	                    x_pre[k] = v2['precision']
	                    x_rec[k] = v2['recall']
	                    x_f1[k] = v2['f1']
	    
	    return x_acc, x_pre, x_rec, x_f1



	def plot_shit(x_sum, name):
	    for key, value in x_sum.items(): 
	        plt.plot(len_list, value, label=key)
	        plt.legend(loc="upper left")
	        plt.title(name)
	    plt.figure()

	    
	name_list = ['accuracy', 'precision', 'recall', 'f1 score']


	x_lists = gen_lists(pa_models)
	len_list = [i for i in range (1, (len(x_lists[0]['PA1'])+1))]

	for i in range (0, len(x_lists)):
		plot_shit(x_lists[i], name_list[i])
	plt.show()

		

