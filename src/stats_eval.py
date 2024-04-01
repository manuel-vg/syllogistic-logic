from numpy import mean, std, array
from pandas import DataFrame, isna
from json import load

#################
# CLASS METRICS #
#################
class Metrics:
    def __init__(self, filename, test_type_2=True):
        self.d = self.__load_json(filename, test_type_2)
        self.types = [k for k in self.d['results'].keys()]

    def __load_json(self, filename, test_type_2):
        f = open(filename, 'r')
        d = load(f)
        f.close()
        if not test_type_2: # remove type 2 from test results
            for k in d.keys():
                if '2' in d[k].keys():
                    del d[k]['2']
        return d

    def __round(self, v, p=100):        
        return round(v*p, 1)

    def __sum(self, c, t=None):
        '''
        returns the sum of row 'c' from the 'results' dictionary/matrix
        such that if 't = -1', then sum of only 'valid' hypotheses
        '''
        return sum([v[c] for k,v in self.d['results'].items() if k != t])

    def __hd(self, t):
        '''
        returns the average Hamming distance (by type 't')
        between the prediction (with necessary premises)
        and the label (real/actual answer)
        '''
        n = sum([v for k, v in self.d['np_hd'][t].items() if k != '0'])
        if n > 0:
            s = sum([int(k)*v for k,v in self.d['np_hd'][t].items()])
            return self.__round(s/n, 1)                
        return 0.0

    def __hd_v(self):
        '''
        returns the average Hamming distance for all (valid) types
        between the prediction (with necessary premises)
        and the label (real/actual answer)
        '''
        # sum of all Hamming distances values
        mp = sum([sum([int(k)*v for k,v in self.d['np_hd'][t].items()]) for t in self.types if t != '-1'])
        if mp > 0:
            mp_n = sum([sum([v for k,v in self.d['np_hd'][t].items() if k != '0']) for t in self.types if t != '-1'])
            hd = self.__round(mp/mp_n, 1)
        else:
            hd = 0.0
        return hd    

    def compute_table(self, show=False, data_frame=True):        
        s = self.__sum # function to sum a column
        r = self.__round # function to get (round) percentages        

        # CREATE ROWS
        # valid (all types)
        valid = [[k, r(v[0]/v[1]), r(v[2]/v[1]), self.__hd(k)] for k,v in self.d['results'].items() if k != '-1']
        # total valid rows
        all_valid = ['Val.', r(s(0, '-1')/s(1, '-1')), r(s(2, '-1')/s(1, '-1')), self.__hd_v()]        
        # invalid
        i = self.d['results']['-1']
        invalid = ['Inv.', r(i[0]/i[1])]
        # total
        total = ['All', r(s(0)/s(1))]
        # concatenate all rows
        table = valid + [all_valid] + [invalid] + [total]

        # CREATE TABLE
        columns=['Inf.', 'Acc.', 'NNM', 'HD']
        df = DataFrame(table, columns=columns) 

        if show:            
            print(df.to_string(index=False))
        
        if data_frame:
            return df

    # TABLE TO DISPLAY THE RESULTS BY TYPE AND BY LENGTH (TL)
    def compute_table_TL(self, unseen=5, show=False, data_frame=True):
        r = self.__round # function to get (round) percentages        
        u_range = [str(u) for u in range(1,unseen+1)]
        table = []

        # CREATE ROWS
        atc = 0
        att = 0
        for t,L in self.d['results'].items(): # for all inference types
            tc = sum([v[0] for v in L.values()])
            tt = sum([v[1] for v in L.values()])
            atc += tc
            att += tt
            row = [t, r(tc/tt)] # accuracy for inferences of type 't'
            for u in u_range: # for all unseen lengths
                c = L[u][0]
                t = L[u][1]                
                row.append(r(c/t)) # accuracy for each (unseen) length 'u' of type 't'
            table.append(row)        
        
        total = ['Val.', r(atc/att)] # total accuracy: all types
        for u in u_range:
            utc = sum([L[u][0] for L in self.d['results'].values()])
            utt = sum([L[u][1] for L in self.d['results'].values()])        
            total.append(r(utc/utt)) # total accuracy by (unseen) length 'u'
        table.append(total)
        
        # CREATE TABLE
        columns = ['Inf.', 'Acc.'] + [f'U{u}' for u in u_range]
        df = DataFrame(table, columns=columns)

        if show:            
            print(df.to_string(index=False))

        if data_frame:
            return df # DataFrame(table)

######################
# MEAN TABLE CLASSES # 
######################
# TABLE TO COMPUTE THE AVERAGE VALUES OF ALL RUNS AND ALL DATASETS
class Table:    
    def __init__(self, tables):
        self.cols = tables[0].columns.values.tolist()[1:]
        self.inf = tables[0]['Inf.'].values.tolist()
        self.table = [array([df[c].values.tolist() for df in tables]).T for c in self.cols]

    def __display_np(self, row):
        if isna(row[0]):
            return '--'
        return round(mean([v for v in row]), 1)
    
    def __display_hd(self, row):
        if isna(row[0]):
            return '--'
        nozr = [v for v in row if v>0]
        if nozr == []:
            return 0.0
        else:
            return round(mean(nozr), 1)
        
    def display_results(self):
        final_table = []
        for i, row in enumerate(self.table[0]):  
            Inf = self.inf[i]
            Best = round(max(row), 1)
            Mean = round(mean(row), 1)
            SD = round(std(row), 1)
            NNM = self.__display_np(self.table[1][i])
            HD = self.__display_hd(self.table[2][i])
            
            final_table.append([Inf, Best, Mean, SD, NNM, HD])
        
        df = DataFrame(final_table, columns = ["Inf.", 'Best', 'Mean', 'SD', 'NNM', 'HD'])
        print(df.to_string(index=False))

# TABLE TO COMPUTE THE AVERAGE VALUES OF ALL RUNS AND ALL DATASETS BY TYPE AND BY LENGTH (TL)
class Table_TL:
    def __init__(self, tables):
        self.cols = tables[0].columns.values.tolist()[1:]
        self.inf = tables[0]['Inf.'].values.tolist()
        self.column_names = tables[0].columns
        self.table = [array([df[c].values.tolist() for df in tables]).T for c in self.cols]        

    def display_results(self):
        final_table = []
        for n,i in enumerate(self.inf):
            row = [i] + [round(mean(self.table[k][n]), 1) for k in range(0, len(self.column_names[1:]))]
            final_table.append(row)
        
        columns = list(self.column_names[:1]) + ['Mean'] + list(self.column_names[2:])
        df = DataFrame(final_table, columns=columns)
        print(df.to_string(index=False))