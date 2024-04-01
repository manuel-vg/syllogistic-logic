import os
from random import Random, shuffle
from itertools import chain
from pandas import DataFrame
from numpy import array, concatenate
from json import load, dump

##################
# ENCODING CLASS #
##################
class Encoding:
    def __init__(self, filename, train_test_split=None):
        # load json file
        self.d = self.__load_json(filename)
        self.name = self.__get_name(filename)
        # retrieve all data from dictionary
        self.vocabulary = self.d['KB']['vocabulary']
        self.constant_name = self.d['KB']['constant_name']
        self.premises = self.d['KB']['premises']
        self.a_chains = self.d['KB']['a_chains']        
        self.inferences = self.d['inferences']
        # initialize/load training/test split dictionaries
        self.first_split = self.__load_split(train_test_split, 0)  # (training split)
        self.second_split = self.__load_split(train_test_split, 1) # (test split)        
        # inferences sorted by length
        self.sbl = None        

    # load JSON dataset
    def __load_json(self, filename):
        f = open(filename, 'r')
        d = load(f)
        f.close()
        return d

    # get JSON dataset filename
    def __get_name(self, filename):
        dataset_name = os.path.basename(filename)
        return os.path.splitext(dataset_name)[0]

    # load/initialize (default) split
    def __load_split(self, train_test_split, i):
        if train_test_split:
            split = self.__load_json(train_test_split[i])
            return {t:[self.inferences[t][p] for p in split[t]] for t in split.keys()}
        return self.__d_types()

    # save split (as indices)
    def __save_split(self):
        # check if "splits" folder exists        
        folder = 'splits'        
        if not os.path.exists(folder):
            os.makedirs(f'{folder}/')        

        # save first and second split as JSON files        
        for split in [(self.first_split, 'train'), (self.second_split, 'test')]:                
            filename = f'{folder}/{self.name}_{split[1]}.json'
            with open(filename, 'w') as file:
                dump({t:[self.inferences[t].index(p) for p in split[0][t]] for t in split[0].keys()}, file)
                file.close()
            print(f'JSON file successfully saved to "{filename}"')
    
    # dictionary of types
    def __d_types(self, v = [[]]):
        '''
        creates a dictionary with the current types as keys 
        and an empty list as values for each key
        '''    
        keys = [k for k in self.inferences.keys()]
        assert len(v) == 1 or len(v) == len(keys), 'Error: the split size is incorrect'
        if len(v) == 1:
            v = [v[0]]*len(keys)
        return dict(zip(keys, v))        

    # SPLIT FUNCTIONS
    def __split_list(self, l, p):    
        newl = list(l)        
        s = round(p*len(newl)) # sample size
        Random(7).shuffle(newl)        

        return newl[:s], newl[s:]
    
    def __types_split(self, split_size):                
        split = self.__d_types(split_size)
        for t, P in self.inferences.items():
            self.first_split[t], self.second_split[t] = self.__split_list(P, split[t])

    def __lengths_split(self, unseen, short_lengths, type_2):
        # sort inferences by length (for each type)
        self.sbl = SortByLength({k:v for k,v in self.inferences.items() if k != '-1'}, self.a_chains, unseen)
        
        # split into training/test datasets (sorted by length) 
        self.sbl.split(short_lengths, type_2)
        
        # collapse (valid) inferences by type (to display info)
        self.first_split = {t:list(chain(*[P for P in L.values()])) for t,L in self.sbl.first_split.items()}
        self.second_split = {t:list(chain(*[P for P in L.values()])) for t,L in self.sbl.second_split.items()}        
        
        # add invalid inferences (split = 20/80)
        self.first_split['-1'], self.second_split['-1'] = self.__split_list(self.inferences['-1'], 0.2)                

    # ENCODING FUNCTIONS
    def __sformula(self, f, sym=False):
        '''
        input: a formula F as string
        output: split F as quantifier, subject and predicate
        '''
        q = f[:1]
        i = f.find(self.constant_name)
        j = f.rfind(self.constant_name)
        subject = f[i:j]
        predicate = f[j:]        
        
        if sym:
            return f'{q}{predicate}{subject}'
        
        return [q, subject, predicate]

    def __one_hot(self, i):
        ohv = [0]*len(self.vocabulary) # one-hot vector 
        ohv[i] = 1
        return ohv

    def __formula_encoder(self, f): 
        '''
        match the words (quantifier, subject, and predicate)
        from the Formula 'f' with Vocabulary and get the index
        to return a one-hot encoding vector
        '''
        q, s, p = [self.vocabulary.index(w) for w in self.__sformula(f)]
        return [self.__one_hot(q), self.__one_hot(s), self.__one_hot(p)] # Encode "tokens" as quantifiers and constants

    def __labels(self, inf_premises, label):
        kb = self.premises
        for f in inf_premises:
            if f not in kb:
                sym_f = self.__sformula(f, sym=True) # symmetric(f) (e.g., symmetric(Eab) = Eba)
                label[kb.index(sym_f)] = 1
            else:
                label[kb.index(f)] = 1
        return label

    def __unfold(self, inf_premises):
        unfolded = []
        for p in inf_premises:
            if p in self.a_chains.keys():
                unfolded += self.a_chains[p]
            else:
                unfolded.append(p)
        return unfolded

    def __encode_proofs(self, proof_dict, e_kb):
        X_encoded_types = {k:[] for k in proof_dict.keys()}
        Y_encoded_types = {k:[] for k in proof_dict.keys()}
        for t, P in proof_dict.items():
            aux_X = []
            aux_Y = []
            for p in P: # for all inferences 'p' in 'P'
                if type(p) == str: # invalid hypotheses
                    h = self.__formula_encoder(p) # encoded (invalid) hypothesis
                    y = [0]*len(self.premises) # label vector of all zeros
                else: # valid hypotheses
                    h = self.__formula_encoder(p[1]) # encoded (valid) hypothesis
                    y = self.__labels(self.__unfold(p[0]), [0]*len(self.premises)) # label vector of needed premises to prove 'h'        
                x = concatenate((e_kb, h)) # input vector (encoded KB + encoded hypothesis)
                aux_X.append(x)
                aux_Y.append(y)            
            X_encoded_types[t] = array(aux_X)
            Y_encoded_types[t] = array(aux_Y)
        
        return X_encoded_types, Y_encoded_types

    # MAIN FUNCTIONS
    def split_dataset(self, split_size=[1], unseen=None, short_lengths=True, type_2=False, save_split=False, summary=True):
        # train/test split
        if unseen:
            self.__lengths_split(unseen, short_lengths, type_2)        
        else:
            self.__types_split(split_size)

        # display split summary
        if summary:
            self.__display_train_test_summary() 

        # save as a 'json' file
        if save_split:
            self.__save_split()
    
    def get_encodings(self):
        # encoded KB (set of premises) as one single vector
        e_kb = list(chain(*[self.__formula_encoder(f) for f in self.premises]))

        # build training/test inputs and labels (X and Y vectors)                
        # TRAINING
        X_train1, Y_train1 = self.__encode_proofs(self.first_split, e_kb)
        # from dictionary (by types) to array
        X_train = array([x for X in X_train1.values() for x in X])
        Y_train = array([y for Y in Y_train1.values() for y in Y])
        
        # TEST
        if self.sbl != None: 
            # create a test set sorted by lenghts
            X_test = {}
            Y_test = {}
            for t,L in self.sbl.second_split.items(): # L=inferences sorted by length
                X_test1, Y_test1 = self.__encode_proofs(L, e_kb)
                X_test[t] = X_test1
                Y_test[t] = Y_test1
        else:
            X_test, Y_test = self.__encode_proofs(self.second_split, e_kb)

        return X_train, X_test, Y_train, Y_test

    # DISPLAY SPLIT INFO FUNCTIONS
    def __roundp(self, v, p=100):
        # return round(v*p, 1)
        return round(v*p)

    def __build_row(self, name, total, split_1, split_2):
        return [name, total, split_1, f'({self.__roundp(split_1/total)})', split_2, f'({self.__roundp(split_2/total)})']

    def __display_train_test_summary(self):  
        # build table to display
        rows = [] # valid types
        for t in [t for t in self.inferences.keys()]:
            total = len(self.inferences[t])
            tr = len(self.first_split[t])
            te = len(self.second_split[t])
            if t == '-1':
                inv = self.__build_row('Inv', total, tr, te)
            else:
                rows.append(self.__build_row(t, total, tr, te))

        # all valid row:        
        valid_all = sum([r[1] for r in rows])
        valid_tr = sum([r[2] for r in rows])
        valid_te = sum([r[4] for r in rows])
        valid = self.__build_row('Val', valid_all, valid_tr, valid_te)

        # total row:
        all_proofs = valid_all + inv[1]
        all_tr = valid_tr + inv[2]
        all_te = valid_te + inv[4]
        total_data = self.__build_row('All', all_proofs, all_tr, all_te)

        # display table
        columns = ['Inf', 'Total', 'Train', '(%)', 'Test', '(%)']
        df = DataFrame(rows + [valid] + [inv] + [total_data], columns=columns)
        
        print('Data splitting ratio:\n' + '-'*21)
        print(df.to_string(index=False)) 

class SortByLength:
    def __init__(self, inferences, a_chains, unseen):
        self.a_chains = a_chains        
        self.unseen = unseen
        self.pbl = self.__sort(inferences) # inferences by length              
        self.first_split = {}
        self.second_split = {}        

    def __length(self, f):
        if f in self.a_chains.keys():
            return len(self.a_chains[f])
        return 1

    def __sort(self, inferences):
        '''
        returns all valid syllogisms sorted by length
        as a dict of types of inferences
        '''
        pbl = {t:{} for t in inferences.keys()} 
        for t in pbl.keys():
            for p in inferences[t]:
                n = sum([self.__length(f) for f in p[0] if f.startswith('A')])
                if n in pbl[t].keys():
                    pbl[t][n].append(p) 
                else:
                    pbl[t][n] = [p]
        return pbl    

    def split(self, short_lengths, type_2):        
        types = self.pbl
        test_lengths = range(1, self.unseen+1)
        
        # if (all) type 2 is included in training
        if type_2:
            types = {k:v for k,v in self.pbl.items() if k != '2'}                        
            self.first_split['2'] = self.pbl['2']
            self.second_split['2'] = {}

        # split training/test by type and by length
        for t,L in types.items():
            if not short_lengths:
                test_lengths = sorted(L.keys())[-self.unseen:]
            self.first_split[t] = {k:v for k,v in L.items() if k not in test_lengths}
            self.second_split[t] = {k:v for k,v in L.items() if k in test_lengths}    