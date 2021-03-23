from node import Node
import pandas as pd

class CART:
    def __init__(self,max_depth=-1,min_leaf_size=1):
        self.max_depth=max_depth
        self.min_leaf_size=min_leaf_size
        self.root=None
        self.dataframe=None
        

    def fit(self,X,y):
        self.dataframe=pd.concat([X, y], ignore_index=True, axis=1)
        self.dataframe.columns=list(X.columns)+list(y.columns)
        self.root=Node(self.dataframe, self.dataframe.index.values, self.max_depth,self.min_leaf_size)

    def print_CART(self):
        print(f'CART info: max_depth={self.max_depth} min_leaf_size={self.min_leaf_size} ')
        self.root.print_node('')


    def predict(self,X):
        if(self.root):
            return self.root.predict(X)
        else:
            print('CART not fitted')


    def score(self, X,y):
        size=len(X.index)
        correct=0
        for index in range(0,size):
            pred_y=self.predict(X.iloc[[index]])
            true_y=y.iat[index,0]
            if(pred_y==true_y):
                correct+=1
        return correct/size




