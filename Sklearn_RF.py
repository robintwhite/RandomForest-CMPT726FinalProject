
# coding: utf-8

# In[ ]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
import preprocessors.HockeyDataSetPreprocessor as HockeyPP

class Sklearn_RF():
    
    def __init__(self,number_of_trees=None, max_depth=None, min_split_size=None, n_features=None):
        
        if number_of_trees == None or number_of_trees == 0:
                number_of_trees = 4
                
        if max_depth==None or max_depth == 0:
            max_depth = None
        
        if min_split_size == None or min_split_size < 2:
            min_split_size = 2
            
        if n_features == None or n_features == 0:
            n_features = None
            
        self.number_of_trees = number_of_trees
        self.max_depth = max_depth
        self.min_split_size = min_split_size
        self.n_features = n_features
        
      
        self.rf = RandomForestClassifier(n_estimators=self.number_of_trees,
                                    max_depth=self.max_depth,
                                    min_samples_split=self.min_split_size,
                                    max_features=self.n_features)
        
    def train(self,data,n_label):
        X = data[:,:-1]
        y = data[:,-1]
        
        self.rf.fit(X,y)
      
    def bagging_predict(self,data):
        
        preds = self.rf.predict(data[:,:-1])
        
        return preds
   
    def evaluate(self,data):
        
        X = data[:,:-1]
        y = data[:,-1]
        
        score = self.rf.score(X,y)
        
        return score
        
    def gridSearch(self,data):
        
        """
         GridSearch sklearn rf to find best params
         n_estimators: number of tress
         max_features: number of features to consider when looking for best split.
         max_depth: max_depth allowed for a tree
         min_samples_split: min number of samples required to split a node
         min_samples_leaf: min umber of samples required to be a leaf node
        """
        X = data[:,:-1]
        
        y = data[:,-1]
        
        rf = RandomForestClassifier()
        params = {'n_estimators':[2,4,8,16,32,64,128],
                  'max_features':['sqrt','log2','auto'],
                  #'max_depth':[2,4,8,16,32,64],
                  #'min_samples_split':[2,4,16,32],
                  #'min_samples_leaf':[2,4,8,16,32,64],
                  'random_state':[1],
                 }

        clf = GridSearchCV(rf,params)
        clf.fit(X, y)
        return  clf.best_score_,clf.best_params_

