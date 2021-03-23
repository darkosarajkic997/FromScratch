import pandas as pd
import numpy as np


WORST_GINI=0.5

class Node:
    def __init__(self, dataframe, indexes,max_depth,min_leaf_size):
        self.dataframe=dataframe
        self.indexes=indexes
        self.max_depth=max_depth
        self.min_leaf_size=min_leaf_size
        self.leaf_label=''
        self.split_atribute=''
        self.split_atribute_type=''
        self.best_gini=WORST_GINI
        self.split_value=None
        self.left_child=None
        self.right_child=None

        if(self.check_end_criteria()):
            self.find_leaf_label()
        else:
            self.find_split()
            left_indexes,right_indexes=self.split_data()
            self.left_child=Node(dataframe,left_indexes,max_depth-1,min_leaf_size)
            self.right_child=Node(dataframe,right_indexes,max_depth-1,min_leaf_size)


    def check_end_criteria(self):
        if(self.max_depth==0):
            return True
        if(len(self.indexes)<self.min_leaf_size):
            return True
        if(self.check_similarty()):
            return True
        return False

    def check_similarty(self):
        class_labels=self.dataframe.loc[self.indexes,self.dataframe.columns[-1]].to_numpy()
        if(np.all(class_labels == class_labels[0])):
            return True
        return False

    def split_data(self):
        tmp_dataframe=self.dataframe.loc[self.indexes,[self.split_atribute,self.dataframe.columns[-1]]]
        if(self.split_atribute_type=='categorical'):
            left_indexes=tmp_dataframe.loc[tmp_dataframe.iloc[:,0]==0].iloc[:,-1].index
            right_indexes=tmp_dataframe.loc[tmp_dataframe.iloc[:,0]==1].iloc[:,-1].index
        else:
            left_indexes=tmp_dataframe.loc[tmp_dataframe.iloc[:,0]<=self.split_value].iloc[:,-1].index
            right_indexes=tmp_dataframe.loc[tmp_dataframe.iloc[:,0]>self.split_value].iloc[:,-1].index
        
        return left_indexes,right_indexes


    def find_leaf_label(self):
        if(self.check_similarty()):
            self.leaf_label=self.dataframe.iat[list(self.dataframe.index).index(self.indexes[0]),-1]
        else:
            values=self.dataframe.loc[self.indexes, self.dataframe.columns[-1]].to_numpy()
            self.leaf_label=np.argmax(np.bincount(values))


    def find_split(self):
        for column in self.dataframe.columns[:-1]:
            atribute_values=self.dataframe.loc[self.indexes,column].to_numpy()
            if(((atribute_values==0) | (atribute_values==1)).all()):
                self.find_binary_gini(column)
            else:
                self.find_gini(column)


    def find_gini(self,column):
        tmp_dataframe=self.dataframe.loc[self.indexes,[column,self.dataframe.columns[-1]]]

        max_value=tmp_dataframe.iloc[:,[0]].max().iloc[0]
        min_value=tmp_dataframe.iloc[:,[0]].min().iloc[0]

        step=(max_value-min_value)/5
        tmp_split_value=min_value+step
        while(tmp_split_value<max_value):
            less_atribute_instances=tmp_dataframe.loc[tmp_dataframe.iloc[:,0]<=tmp_split_value].iloc[:,-1].to_numpy()
            more_atribute_instances=tmp_dataframe.loc[tmp_dataframe.iloc[:,0]>tmp_split_value].iloc[:,-1].to_numpy()
            
            less_number=less_atribute_instances.size
            more_number=more_atribute_instances.size
            total_num=less_number+more_number

            gini_value=(1-self.calcuate_sum_for_gini(less_atribute_instances))*(less_number/total_num)+(1-self.calcuate_sum_for_gini(more_atribute_instances))*(more_number/total_num)
            if(self.best_gini>gini_value):
                self.best_gini=gini_value
                self.split_atribute=column
                self.split_atribute_type='numerical'
                self.split_value=tmp_split_value

            tmp_split_value+=step




    def find_binary_gini(self,column):
        tmp_dataframe=self.dataframe.loc[self.indexes,[column,self.dataframe.columns[-1]]]

        zero_atribute_instances=tmp_dataframe.loc[tmp_dataframe.iloc[:,0]==0].iloc[:,-1].to_numpy()
        one_atribute_instances=tmp_dataframe.loc[tmp_dataframe.iloc[:,0]==1].iloc[:,-1].to_numpy()

        zero_number=zero_atribute_instances.size
        one_number=one_atribute_instances.size
        total_num=zero_number+one_number

        gini_value=(1-self.calcuate_sum_for_gini(zero_atribute_instances))*(zero_number/total_num)+(1-self.calcuate_sum_for_gini(one_atribute_instances))*(one_number/total_num)
        if(self.best_gini>gini_value):
            self.best_gini=gini_value
            self.split_atribute=column
            self.split_atribute_type='categorical'
            self.split_value=None
            
            

    def calcuate_sum_for_gini(self,list_of_instances):
        probablilties=[]
        size=list_of_instances.size
        unique, counts = np.unique(list_of_instances, return_counts=True)
        for number in np.asarray((unique, counts)).T:
            probablilties.append(number[1]/size)

        sum_of_squares=np.sum(np.square(np.asarray(probablilties)))
        return sum_of_squares 
        


    def print_node(self,indent):
        
        if(self.leaf_label!=''):
            print(indent+f'[Leaf: {self.leaf_label}]')
            indent+='|--------'
        elif(self.split_atribute_type=='categorical'):
            print(indent+f'GI:{self.best_gini} ATR:{self.split_atribute}')
            indent+='|--------'
            self.right_child.print_node(indent)
            self.left_child.print_node(indent)
        else:
            print(indent+f'GI:{self.best_gini} ATR:{self.split_atribute} VAL:{self.split_value}')
            indent+='|--------'
            self.right_child.print_node(indent)
            self.left_child.print_node(indent)


    def predict(self,X):
        if(self.leaf_label!=''):
            return self.leaf_label
        else:
            value=X.loc[X.index[0],self.split_atribute]
            if(self.split_atribute_type=='categorical'):
                if(value==0):
                    return self.left_child.predict(X)
                else:
                    return self.right_child.predict(X)
            else:
                if(value<=self.split_value):
                    return self.left_child.predict(X)
                else:
                    return self.right_child.predict(X)


    

