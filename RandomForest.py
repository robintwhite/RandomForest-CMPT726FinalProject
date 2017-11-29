from Tree import Tree
from multiprocessing import Pool
from multiprocessing import Manager
from functools import partial

class RandomForest():
    """
    Class to hold logic for orchestrating learning and predicting with multiple trees.

    """


    def __init__(self, number_of_trees, max_depth, min_split_size, n_features, workers):
        """
        Initialize instance of a RandomForest.

        @param number_of_trees - number of trees to create for the forest.
        @param max_depth - the maximum depth of the trees in the forest.
        @param min_split_size - the minimum number of samples required at a node to allow for further splitting.
        @param n_features - the number of features to use when building each tree in the random forest.

        """
        self.workers = workers
        self.trees = []

        for value in range(number_of_trees):
            self.trees.append(Tree(value, max_depth, min_split_size, n_features))

        # TODO: Remove this printing stuff since it's just a placeholder.
        for tree in self.trees:
            tree.printID()

    def _build_tree(self, tree, train_data, target_class, trained_trees, splitf):
        """
        Helper function to use for multi-processing trees during training.

        @param tree - the tree to build.
        @param train_data - the training data to train each tree in the forest on.
        @param target_class - the target class we want to predict using the random forest.
        @param trained_trees - shared multiprocessing.Manager().list() to return trained trees.

        IMPORTANT: Using the multiprocess version currently will result in lower accuracy, but that's okay since this
                   enhancement is mostly to speed up developement.

        """
        tree.tree_build_util(train_data, target_class,splitf)
        trained_trees.append(tree)

    #Specify split function (splitf) parameter for classification or regression: gini, entropy, variance
    def train(self, train_data, target_class,splitf):
        """
        Train the random forest on the given data.

        @param train_data - information for the training data.
        @param target_class - column we want to be able to predict.

        """
        if self.workers:
            trained_trees = Manager().list()
            partial_function = partial(self._build_tree, train_data=train_data, target_class=target_class,  trained_trees=trained_trees,splitf=splitf)
            with Pool(processes=self.workers) as workers:
                workers.map(partial_function, self.trees)
            self.trees = trained_trees
        else:
            for tree in self.trees:
                tree.tree_build_util(train_data, target_class,splitf)


    def bagging_predict(self, test_data):
        """
        Predict the values in the given data using the parameters learned by the random forest.

        @param test_data_x - feature information for the test data.
        @param trees - trees built in train

        """
        print("Predicting on the given dataset!")

        test_data_x = test_data[:,:-1]
        test_data_y = test_data[:,-1]
        predictions = []
        # TODO: Add logic here.
        for row in test_data_x:
            #for each test case, majority vote for trees
            prediction = [tree.predict(tree.root, row) for tree in self.trees] #array with prediction from each tree
            predictions.append(max(set(prediction), key=prediction.count)) #Majority vote and store prediction

        return predictions

    def evaluate(self, predictions, test_data_y):
        #check prediction of each row against test data test_data_y = test_data[:,-1]
        correct = 0
        for i in range(len(test_data_y)):
            if test_data_y[i] == predictions[i]:
                correct += 1
        return correct / float(len(test_data_y)) * 100.0
