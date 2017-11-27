
# coding: utf-8

# In[ ]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
import preprocessors.HockeyDataSetPreprocessor as HockeyPP

class Sklearn_RF():
    
    def __init__(self,number_of_trees=0, max_depth=0, min_split_size=0, n_features=0):
        self.number_of_trees = number_of_trees
        self.max_depth = max_depth
        self.min_split_size = min_split_size
        self.n_features = n_features
        self.rf =  rf = RandomForestClassifier(n_estimators=self.number_of_trees,
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
        
        score = self.rf.evaluate(X,y)
        
        return score
        
    def gridSearch(X,y):
        
        """
         GridSearch sklearn rf to find best params
         n_estimators: number of tress
         max_features: number of features to consider when looking for best split.
         max_depth: max_depth allowed for a tree
         min_samples_split: min number of samples required to split a node
         min_samples_leaf: min umber of samples required to be a leaf node
        """
        
        rf = RandomForestClassifier()
        params = {'n_estimators':[2,4,8,16,32,64,128],
                  'max_features':['sqrt','log2','auto'],
                  'max_depth':[2,4,8,16,32,64],
                  'min_samples_split':[2,4,16,32],
                  'min_samples_leaf':[2,4,8,16,32,64],
                  'random_state':[1],
                 }

        clf = GridSearchCV(rf,params)
        clf.fit(X, y)
        print(clf.cv_results_)

