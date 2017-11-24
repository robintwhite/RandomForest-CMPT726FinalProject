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
            self.trees.append(Tree(value)) #What params are given to Tree here then?

        # TODO: Remove this printing stuff since it's just a placeholder.
        for tree in self.trees:
            tree.printID()

    def train(self, train_data):
        """
        Train the random forest on the given data.

        @param train_data - feature information for the training data.

        """
        # TODO: Train self.trees

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
