from Tree import Tree


class RandomForest():
    """
    Class to hold logic for orchestrating learning and predicting with multiple trees.

    """

    def __init__(self, number_of_trees):
        """
        Initialize instance of a RandomForest.

        """
        
        self.number_of_trees = number_of_trees

    def train(self, train_data, number_of_trees):
        """
        Train the random forest on the given data.

        @param train_data - feature information for the training data.
        @param number_of_trees - number of trees to create for the forest.

        """
        trees = []

        for value in range(number_of_trees):
            trees.append(Tree(value, train_data, max_depth, min_size, n_features))

        # TODO: Remove this printing stuff since it's just a placeholder.
        for tree in trees:
            tree.printID()
        print("Training on the given dataset!")

        return trees


    def predict(self, test_data, trees):
        """
        Predict the values in the given data using the parameters learned by the random forest.

        @param test_data_x - feature information for the test data.
        @param trees - trees built in train

        """
        print("Predicting on the given dataset!")

        test_data_x = test_data[:,:-1]
        test_data_y = test_data[:,-1]

        # TODO: Add logic here.
