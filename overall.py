'''
OVERALL PERFORMANCE 
-------------------
Examples: (from a terminal)
1) To train and evaluate CNNs (all 5 datasets):

	python -c 'import overall; overall.train_and_evaluate_models(model_type="CNN")'

2) To train and evaluate RNNs (only 1st and 3rd dataset):
	
	python -c 'import overall; overall.train_and_evaluate_models(model_type="RNN", datasets=[1,3])'	

3) To display final results for transformers (as an average table of all runs and all datasets):

	python -c 'import overall; overall.display_results(model_type="TRA")'
'''
# import libraries
import sys
sys.path.insert(0, 'src') # include the source code folder (src) path in the current directory

from encoding import Encoding 
from neural_models import Neural_Model
from stats_eval import Metrics, Table
from json import load

def train_and_evaluate_models(model_type, datasets=[1,2,3,4,5]):
	# Load hyper parameters and arguments
	f = open('src/h_parameters.json', 'r')
	params = load(f)
	f.close()
	args = params[model_type] # select 'args' by type of model	

	for i in datasets: # 5 Datasets
		# load dataset
		print(f'\nDATASET {i}:\n' + '-'*10)
		ds = f'datasets/ds{i}.json'		
		# splitting into training and test data	
		split = [0.2, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75] # types = [-1, 1, 2, 3, 4, 5, 6, 7]
		en = Encoding(ds)	
		en.split_dataset(split_size=split, summary=True)	
		X_train, X_test, Y_train, Y_test = en.get_encodings()		
		# training data
		print('\nTraining and evaluation process:\n' + '-'*32)
		for r in range(1,4): # 3 runs
			name = f'exp1_overall_{model_type.lower()}_ds{i}_run{r}'
			model = Neural_Model(args, name, X_train, Y_train)
			model.build(model_type=model_type)
			model.fit()
			# test data
			model.evaluate(X_test, Y_test)	

def display_results(model_type, datasets=[1,2,3,4,5]):
	tables = [] # list to save all results
	for i in datasets: # 5 Datasets
		for r in range(1,4): # 3 runs
			filename = f'results/exp1_overall_{model_type.lower()}_ds{i}_run{r}.json'
			r = Metrics(filename)
			tables.append(r.compute_table())

	final_table = Table(tables)
	final_table.display_results()