'''
UNSEEN LENGTHS FOR SHORT/LONG INFERENCES
----------------------------------------
Examples: (from a terminal)
1) To train and evaluate MLPs for short inferences:

	python -c 'import unseen; unseen.train_and_evaluate_models(model_type="MLP")'

2) To train and evaluate transformers for long inferences:
	
	python -c 'import unseen; unseen.train_and_evaluate_models(model_type="TRA", short_lengths=False)'

3) To train and evaluate RNNs for long inferences with (all) inferences of type 2 in training:

	python -c 'import unseen; unseen.train_and_evaluate_models(model_type="RNN", short_lengths=False, type_2=True)'

4) To display the final results for CNNs for short inferences with (all) lengths of type 2 in training:

	python -c 'import unseen; unseen.display_results(model_type="CNN", type_2=True)'

5) To display the final results for transformers for short inferences without inferences of type 2 for test:

	python -c 'import unseen; unseen.display_results(model_type="TRA", test_type_2=False)'
'''
# import libraries
import sys
sys.path.insert(0, 'src') # include the source code folder (src) path in the current directory

from encoding import Encoding 
from neural_models import Neural_Model
from stats_eval import Metrics, Table_TL
from json import load

def get_name(sl, t2):
	name = {(1,0):'exp2', (0,0):'exp3', (1,1):'exp4', (0,1):'exp5'}
	return name[(int(sl), int(t2))]

def train_and_evaluate_models(model_type, datasets=[1,2,3,4,5], short_lengths=True, type_2=False):
	# Load hyper parameters and arguments
	f = open('src/h_parameters.json', 'r')
	params = load(f)
	f.close()
	args = params[model_type] # select 'args' by type of model

	# experiment type: short/long unseen inferences with/without type 2 in the training dataset
	exp_type = get_name(short_lengths, type_2)

	for i in datasets: # 5 Datasets
		# load dataset
		print(f'\nDATASET {i}:\n' + '-'*10)
		ds = f'datasets/ds{i}.json'		
		# splitting into training and test data				
		en = Encoding(ds)		
		en.split_dataset(unseen=5, short_lengths=short_lengths, type_2=type_2)
		X_train, X_test, Y_train, Y_test = en.get_encodings()		
		# training data
		print('\nTraining and evaluation process:\n' + '-'*32)
		for r in range(1,4): # 3 runs
			name = f'{exp_type}_unseen_{model_type.lower()}_ds{i}_run{r}'
			model = Neural_Model(args, name, X_train, Y_train)
			model.build(model_type=model_type)
			model.fit()			
			# test data
			model.evaluate(X_test, Y_test, by_length=True)

def display_results(model_type, datasets=[1,2,3,4,5], short_lengths=True, type_2=False, test_type_2=True):
	exp_type = get_name(short_lengths, type_2)
	tables = [] # list to save all results
	for i in datasets: # 5 Datasets
		for r in range(1,4): # 3 runs
			filename = f'results/{exp_type}_unseen_{model_type.lower()}_ds{i}_run{r}.json'
			r = Metrics(filename, test_type_2)
			tables.append(r.compute_table_TL())

	final_table = Table_TL(tables)
	final_table.display_results()