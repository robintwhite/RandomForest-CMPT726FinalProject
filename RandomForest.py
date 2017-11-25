from Tree import Tree


class RandomForest():
    """
    Class to hold logic for orchestrating learning and predicting with multiple trees.

    """


    def __init__(self, number_of_trees, max_depth, min_split_size, n_features):
        """
        Initialize instance of a RandomForest.

        @param number_of_trees - number of trees to create for the forest.
        @param max_depth - the maximum depth of the trees in the forest.
        @param min_split_size - the minimum number of samples required at a node to allow for further splitting.
        @param n_features - the number of features to use when building each tree in the random forest.

        """
        self.trees = []

        for value in range(number_of_trees):
            self.trees.append(Tree(value, max_depth, min_split_size, n_features))

        # TODO: Remove this printing stuff since it's just a placeholder.
        for tree in self.trees:
            tree.printID()


    def train(self, train_data, target_class):
        """
        Train the random forest on the given data.

        @param train_data - information for the training data.
        @param target_class - column we want to be able to predict.

        """
        for tree in self.trees:
            tree.tree_build_util(train_data, target_class)


    def bagging_predict(self, test_data):
        """
        Predict the values in the given data using the parameters learned by the random forest.

        @param test_data_x - feature information for the test data.
        @param trees - trees built in train

        """
        print("Predicting on the given dataset!")

        test_data_x = test_data[:,:-1]
        test_data_y = test_data[:,-1]
        idx = 0
        # TODO: Add logic here.
        for row in test_data_x:
            #for each test case, majority vote for trees
            prediction = [tree.predict(tree.root, row) for tree in self.trees] #array with prediction from each tree
            predictions[idx] = max(set(prediction), key=prediction.count) #Majority vote and store prediction
            idx += 1

        return predictions

    def evaluate(self, predictions, test_data_y):
        #check prediction of each row against test data test_data_y = test_data[:,-1]
        pass
