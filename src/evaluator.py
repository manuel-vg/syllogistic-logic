import os
from numpy import round
from json import dump

# EVALUATE TEST DATA AND SAVE RESULTS
class Evaluator:
    def __init__(self, model, X_test, Y_test):
        self.model = model
        self.X_test = X_test
        self.Y_test  = Y_test

    def __save_evaluation(self, results, np_hd={}, folder='results'):
        # check if folder exists        
        if not os.path.exists(folder):
            os.makedirs(f'{folder}/')        
        # create a path filename
        filename = f'{folder}/{self.model.name}.json'
        
        # save the model as JSON
        evaluation = {'results': results, 
                      'np_hd': np_hd}
        
        with open(filename, 'w') as file:
            dump(evaluation, file)
            file.close()

        print(f'Evaluation was successfully saved to "{filename}"')

    def __error_metrics(self, actual_label, prediction):
        m = 0
        a = 0
        for i, j in zip(actual_label, prediction):
            if not(i == j):
                if i == 0:
                    a += 1 # additional premise
                else:
                    m += 1 # missing premise                    
        return [m + a, m]

    def __plus_one(self, d, k):
        if k in d.keys():
            d[k] += 1
        else:
            d[k] = 1
        return d

    def evaluate(self):
        types = [t for t in self.X_test.keys() if self.X_test[t].shape != (0,)] # list of current types to test from 'X_test'
        results = {t:[0,0,0] for t in types} 
        np_hd = {t:{} for t in types if t != '-1'}

        for t in types:
            all_pred = self.model.predict(self.X_test[t], verbose=0)
            n = len(all_pred) 
            results[t][1] = n 
            for i in range(n):
                # Comparison between the 'prediction' and the 'actual label'                
                prediction = round(all_pred[i])                
                actual_label = self.Y_test[t][i]
                correct = all(prediction == actual_label)
                if correct:
                    results[t][0] += 1
                
                # Not necessarily minimal (NNM) inferences
                hd, mp = self.__error_metrics(actual_label, prediction)                                    
                # If no missing premises (necessary premises):
                if mp == 0 and t != '-1':
                    results[t][2] += 1    
                    if not correct:
                        # Hamming distance between the "necessary premises" prediction and the actual label
                        np_hd[t].update(self.__plus_one(np_hd[t], hd))
        
        # Save evaluation to a JSON file
        self.__save_evaluation(results, np_hd)

    def evaluate_by_lenght(self, unseen=5):
        types = [t for t in self.X_test.keys() if self.X_test[t] != {}] # list of current types to test from 'X_test'
        # set dict for results of unseen long/short lengths (1-5) expermients
        results = {t:{u:[0,0] for u in range(1,unseen+1)} for t in types}
        
        for t in types:
            u_keys = sorted(list(self.X_test[t].keys()))
            for k in u_keys:
                u = u_keys.index(k) + 1 # long/short unseen lengths
                all_pred = self.model.predict(self.X_test[t][k], verbose=0)
                n = len(all_pred)
                results[t][u][1] = n
                for i in range(n):
                    # Comparison between the 'prediction' and the 'actual label'
                    results[t][u][0] += all(round(all_pred[i]) == self.Y_test[t][k][i])

        # Save evaluation to a JSON file
        self.__save_evaluation(results)