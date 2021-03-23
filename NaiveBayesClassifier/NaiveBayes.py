import pandas as pd
import numpy as np
from math import sqrt
from math import pi
from math import exp

class NaiveBayesClassifier:

    def __init__(self,X,y):
        self.dataframe=pd.concat([X, y], ignore_index=True, axis=1)
        self.dataframe.columns=list(X.columns)+['target']
        self.number_rows=len(self.dataframe.index)
        self.classes=None
        self.class_dict={}
        self.stat_dict={}
        self.split_dataframe()
        self.calcualte_summary_for_all_classes()


    def split_dataframe(self):
        self.classes=self.dataframe.target.unique()
        grouped=self.dataframe.groupby('target')
        self.class_dict={}
        for value in self.classes:
            self.class_dict[value]= grouped.get_group(value)

    def mean(self,dataframe_column):
        return np.mean(dataframe_column.to_numpy())


    def std_deviation(self, dataframe_column):
         return np.std(dataframe_column.to_numpy())

    
    def summarize_dataframe(self,dataframe):
        stats={}
        for column in list(dataframe.columns)[:-1]:
            column_stat=[]
            column_stat.append(self.mean(dataframe[column]))
            column_stat.append(self.std_deviation(dataframe[column]))
            column_stat.append(dataframe[column].count())
            stats[column]=column_stat
        
        return stats


    def calcualte_summary_for_all_classes(self):
        self.stat_dict={}
        for key, dataframe in self.class_dict.items():
            self.stat_dict[key]=self.summarize_dataframe(dataframe)


    def calculate_probability(self,x, mean, stdev):
	    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	    return (1 / (sqrt(2 * pi) * stdev)) * exponent



    def predict(self,x):
        max=0
        max_class=0
        for value in self.classes:
            stats=self.stat_dict[value]
            prob=1
            count=0
            for column in list(self.dataframe.columns)[:-1]:
                column_stats=stats[column]
                value_x=x[column]
                prob*=self.calculate_probability(value_x,column_stats[0],column_stats[1])
                count_val=column_stats[2]

            prob*=(count_val/self.number_rows)

            if(prob>max):
                max=prob
                max_class=value

        return max_class


    def score(self,X_test,y_test):
        index=0
        correct=0
        while(index<len(X_test.index)):
            prediction=self.predict(X_test.iloc[index])
           
            if(prediction==y_test.iloc[index]):
                correct+=1
            
            index+=1

        print(correct/len(X_test.index))

        return correct/len(X_test.index)