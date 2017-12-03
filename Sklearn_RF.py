from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from sklearn import datasets
import preprocessors.HockeyDataSetPreprocessor as HockeyPP
from matplotlib import pyplot as plt
import numpy as np

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


    def train(self,data,n_label):
        if n_label == 'GP_greater_than_0':
            self.rf = RandomForestClassifier(n_estimators=self.number_of_trees,
                                        max_depth=self.max_depth,
                                        min_samples_split=self.min_split_size,
                                        max_features=self.n_features)
        else:
            self.rf = RandomForestRegressor(n_estimators=self.number_of_trees,
                                    max_depth=self.max_depth,
                                    min_samples_split=self.min_split_size,
                                    max_features=self.n_features)
        X = data[:,:-1]
        y = data[:,-1]

        self.rf.fit(X,y)

    def bagging_predict(self,data):

        preds = self.rf.predict(data[:,:-1])

        return preds

    def evaluate(self,data, tree_type):

        X = data[:,:-1]
        y = data[:,-1]
        score = 0

        if tree_type == 'classifier':
            score = self.rf.score(X,y)

        elif tree_type == 'regressor':
            preds = self.rf.predict(data[:,:-1])
            score = mean_squared_error(y,preds)

        return score

    def gridSearch(self,train_data,test_data,tree_type='classifier'):

        """
         GridSearch sklearn rf to find best params
         n_estimators: number of tress
         max_features: number of features to consider when looking for best split.
         max_depth: max_depth allowed for a tree
         min_samples_split: min number of samples required to split a node
         min_samples_leaf: min umber of samples required to be a leaf node
        """
        X_train = train_data[:,:-1]

        y_train = train_data[:,-1]

        X_test = test_data[:,:-1]

        y_test = test_data[:,-1]

        if tree_type == 'classifier':

            rf = RandomForestClassifier(n_estimators=self.number_of_trees,
                                    max_depth=self.max_depth,
                                    min_samples_split=self.min_split_size,
                                    max_features=self.n_features)

        elif tree_type == 'regressor':
            rf = RandomForestRegressor(n_estimators=self.number_of_trees,
                                    max_depth=self.max_depth,
                                    min_samples_split=self.min_split_size,
                                    max_features=self.n_features)

        params = {'n_estimators':[2,4,8,16,32,64,128],
                  #'max_features':['sqrt','log2','auto'],
                  #'max_depth':[2,4,8,16,32,64],
                  #'min_samples_split':[2,4,16,32],
                  #'min_samples_leaf':[2,4,8,16,32,64],
                  'random_state':[1],
                 }

        if tree_type == 'regressor':
             clf = GridSearchCV(rf,params,scoring='neg_mean_squared_error',return_train_score=True)
        else:
            clf = GridSearchCV(rf,params,return_train_score=True)

        clf.fit(X_train, y_train)

        print("Parameters for the best {} estimator on the training set:".format(tree_type))

        print()

        print(clf.best_params_)

        print()

        print("cross-validation accuracy scores:")

        means = clf.cv_results_["mean_test_score"]

        stds = clf.cv_results_['std_test_score']

        for mean, std, param in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, param))

        print()

        print("Test set Score:")

        print()

        y_true, y_pred = y_test, clf.predict(X_test)

        if tree_type == 'classifier':
            print(classification_report(y_true,y_pred))

        elif tree_type == 'regressor':
            print("Total mean_squared_error: ",mean_squared_error(y_true,y_pred))

        res = clf.cv_results_

        #generate plots
        plt.figure(figsize=(12,12))

        plt.title("GridSearchCV multiple number of estimators",fontsize=16)
        plt.xlabel('Number of estimators',fontsize=14)

        y_label = "Accuracy"

        if tree_type == 'regressor':
            y_label = "Accuracy ({})".format("neg_mean_squared_error")

        elif tree_type == 'classifier':
            y_label = "Accuracy ({})".format("accuracy_classification_score")

        plt.ylabel(y_label,fontsize=14)
        plt.grid()

        ax = plt.axes()

        x_axis = np.array(res['param_n_estimators'].data)

        plt.plot(x_axis,res['mean_train_score'],label='train')

        plt.plot(x_axis,res['mean_test_score'],label='validation')
        plt.legend(fontsize=12)
        #plt.show()
        filename = 'sklearn_rf_{}_cv.png'.format(tree_type)
        plt.savefig(filename)

        return  clf.cv_results_
