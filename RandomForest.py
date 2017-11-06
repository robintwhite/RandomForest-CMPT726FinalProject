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
            self.trees.append(Tree(value))

        # TODO: Remove this printing stuff since it's just a placeholder.
        for tree in self.trees:
            tree.printID()


    def train(self, train_data_x, train_data_y):
        """
        Train the random forest on the given data.

        @param train_data_x - feature information for the training data.
        @param train_data_y - associated labels for each datapoint in train_data_x.

        """
        print("Training on the given dataset!")

        # TODO: Add logic here.


    def predict(self, test_data_x):
        """
        Predict the values in the given data using the parameters learned by the random forest.

        @param test_data_x - feature information for the test data.

        """
        print("Predicting on the given dataset!")

        # TODO: Add logic here.
