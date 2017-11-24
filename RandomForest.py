from Tree import Tree


class RandomForest():
    """
    Class to hold logic for orchestrating learning and predicting with multiple trees.

    """


    def __init__(self, number_of_trees):
        """
        Initialize instance of a RandomForest.

        @param number_of_trees - number of trees to create for the forest.

        """
        self.trees = []

        for value in range(number_of_trees):
            self.trees.append(Tree(value)) #Still don't know what to give here, just id then?

        # TODO: Remove this printing stuff since it's just a placeholder.
        for tree in self.trees:
            tree.printID()

    def train(self, train_data, max_depth, min_size, n_features):
        """
        Train the random forest on the given data.

        @param train_data - feature information for the training data.

        """
        # Make set of trees
        
        for tree in self.trees:
            tree.tree_build_util(train_data, max_depth, min_size, n_features)

    def predict(self, test_data):
        """
        Predict the values in the given data using the parameters learned by the random forest.

        @param test_data_x - feature information for the test data.
        @param trees - trees built in train

        """
        print("Predicting on the given dataset!")

        test_data_x = test_data[:,:-1]
        test_data_y = test_data[:,-1]

        # TODO: Add logic here.
        for tree in self.trees:
            predictions = tree.predict(test_data_x)
            
            #TODO: Aggregating voting logic here
            #prediction = max(set(predictions, key=predictions.count)) #Maybe?
        
        return prediction
