import json
import matplotlib.pyplot as plt



with open("sgd_metrics8000.json") as f :
	metrics8 = json.load(f)
	
with open("sgd_metrics10000.json") as f :
	metrics10 = json.load(f)

len_x = [i for i in range(1,81)]

def plotstuff10(batch, metric, x_len):
	for model,scores in batch.items() :
		plt.plot(x_len, scores[metric], label=model)
	plt.legend(loc="upper left")
	plt.title(f" {metric} - 10000 batch")
	plt.show()
		
def plotstuff8(batch, metric, x_len):
	for model,scores in batch.items() :
		plt.plot(x_len, scores[metric], label=model)
	plt.legend(loc="upper left")
	plt.title(f" {metric} - 8000 batch")
	plt.show()
	
def plotstuff(batch1, batch2, model, metric, x_len):
	b1 = batch1[model][metric]
	b2 = batch2[model][metric]
	plt.plot(x_len, b1, label = "8000")
	plt.plot(x_len, b2, label = "10000")
	plt.legend(loc = "upper left")
	plt.title(f" {metric} - batch comparison ")
	plt.show()
	
	
plotstuff(metrics8, metrics10, "SGD1", "precision", len_x)
plotstuff(metrics8, metrics10, "SGD1", "recall", len_x)
plotstuff(metrics8, metrics10, "SGD1", "f1", len_x)
	
"""	
plotstuff8(metrics8, "accuracy", len_x)
plotstuff8(metrics8, "recall", len_x)
plotstuff8(metrics8, "precision", len_x)
plotstuff8(metrics8, "f1", len_x)

plotstuff10(metrics10, "accuracy", len_x)
plotstuff10(metrics10, "recall", len_x)
plotstuff10(metrics10, "precision", len_x)
plotstuff10(metrics10, "f1", len_x)
"""
"""	
def gen_lists(perc_models):
	    x_acc, x_pre, x_rec, x_f1 = {}, {}, {}, {}
	    for k,v in perc_models.items():
	        for k2, v2 in v.items():
	            
	    
	    return x_acc, x_pre, x_rec, x_f1
	    
	def plot_shit(x_sum, name):
	    for key, value in x_sum.items(): 
	        plt.plot(len_list, value, label=key)
	        plt.legend(loc="upper left")
	        plt.title(name)
	    plt.figure()

	    
	name_list = ['accuracy', 'precision', 'recall', 'f1 score']


	x_lists = gen_lists(perc_models)
	len_list = [i for i in range (1, (len(x_lists[0]['PA1'])+1))]

	for i in range (0, len(x_lists)):
		plot_shit(x_lists[i], name_list[i])
	plt.show()
"""
