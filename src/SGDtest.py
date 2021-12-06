#spark dependencies
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
#data cleaning dependencies
import json
import nltk
import text_clean as clean
# learning dependencies
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import joblib
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix


#function to make sure only a single sparksession in global space
def getSparkSessionInstance(sparkConf):
	if ("sparkSessionSingletonInstance" not in globals()):
		globals()["sparkSessionSingletonInstance"] = SparkSession\
			.builder\
			.config(conf=sparkConf)\
			.getOrCreate()
	return globals()["sparkSessionSingletonInstance"]

# main function to read, clean, preprocess and evaluate scores
def process(time, rdd):
	if not rdd.isEmpty():
		print(f"======== {time} ========")
	
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
		

		# collect lemmatized text & target as ndarray
		sentiment = np.array([int(row['target']) for row in df.collect()])
		text = np.array([str(row['lemmatized_text']) for row in df.collect()])
		
		print(f"batch size : {len(text)}")
		
		# vectorize cleaned text using hashing vectorizer
		X_test = vectorizer.transform(text)
		
		
		
		
		# predict and get various performance metrics
		for cls_name, cls in sgd_models.items():
			pred = cls["model"].predict(X_test)
			cls["scores"]["accuracy"].append(round(accuracy_score(sentiment, pred), 4))
			cls["scores"]["precision"].append(round(precision_score(sentiment, pred), 4))
			cls["scores"]["recall"].append(round(recall_score(sentiment, pred), 4))
			cls["scores"]["f1"].append(round(f1_score(sentiment, pred), 4))
			#print(f"{cls_name} : {cls['scores']}")
		
		
		

if __name__ == '__main__':
	#create sparkcontext and the streaming context
	sc = SparkContext("local[2]", "recieveData")
	ssc = StreamingContext(sc, 1)

	#read the socket data 
	lines = ssc.socketTextStream("localhost", 6100)

	vectorizer = HashingVectorizer( decode_error='ignore', n_features=2**20, alternate_sign=False )
	
	sgd_models = {
		"SGD1" : {
			"model" : None,
			"scores" : {
				"accuracy" : [],
				"precision" : [],
				"recall" : [],
				"f1" :  [],
			}
		},
		"SGD2" : {
			"model" : None,
			"scores" : {
				"accuracy" : [],
				"precision" : [],
				"recall" : [],
				"f1" :  [],
			}
		},
		"SGD3" : {
			"model" : None,
			"scores" : {
				"accuracy" : [],
				"precision" : [],
				"recall" : [],
				"f1" :  [],
			}
		},
		"SGD4" : {
			"model" : None,
			"scores" : {
				"accuracy" : [],
				"precision" : [],
				"recall" : [],
				"f1" :  [],
			}
		}
	}
	
	for cls_name in sgd_models :
		sgd_models[cls_name]["model"] = joblib.load(f"{cls_name}.pkl")
		
	

	lines.foreachRDD(process)



	ssc.start()
	ssc.awaitTermination(timeout=60*8)
	
	ssc.stop(stopGraceFully=True)
	
	print("all scores calculated")
	
	export_dict = {
		"SGD1" : sgd_models["SGD1"]['scores'],
		"SGD2" : sgd_models["SGD2"]['scores'],
		"SGD3" : sgd_models["SGD3"]['scores'],
		"SGD4" : sgd_models["SGD4"]['scores'],
	}
	
	with open("sgd_metrics.json", "w") as outfile:
		json.dump(export_dict, outfile)
		
	print("all metrics saved")
	
	
