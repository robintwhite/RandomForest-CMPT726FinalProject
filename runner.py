"""
Script to help run the RandomForest program.

"""
from RandomForest import RandomForest
from argparse import ArgumentParser

import preprocessors.HockeyDataSetPreprocessor as HockeyPP
import preprocessors.BreastCancerDataSetPreprocessor as BreastCancerPP



def main():
    argument_parser = ArgumentParser(
        description="Script to run the RandomForest program.",
        add_help=False)
    mutually_exclusive_group = argument_parser.add_mutually_exclusive_group()
    mutually_exclusive_group.add_argument(
        '--use_gini',
        action='store_true',
        help="Use the Gini index for attribute splitting in the decision trees.")
    mutually_exclusive_group.add_argument(
        '--use_entropy',
        action='store_true',
        help="Use entropy for attribute splitting in the decision trees.")
    mutually_exclusive_group2 = argument_parser.add_mutually_exclusive_group()
    mutually_exclusive_group2.add_argument(
        '--use_hockey_preprocessor',
        action='store_true',
        help="Use hockey dataset preprocessing logic on the given dataset.")
    mutually_exclusive_group2.add_argument(
        '--use_breast_cancer_preprocessor',
        action='store_true',
        help="Use breast cancer dataset preprocessing logic on the given dataset.")
    argument_parser.add_argument(
        '-d', '--data_file',
        required=True,
        help="File containing the dataset.")
    argument_parser.add_argument(
        '-t', '--number_of_trees',
        default=4,
        help="The number of trees to create for the random forest.")
    argument_parser.add_argument(
        '-h', '--help',
        action='help',
        help="Show this message and exit.")
    arguments = argument_parser.parse_args()

    dataset_file = arguments.data_file

    preprocessor = None

    if arguments.use_hockey_preprocessor:
        preprocessor = HockeyPP
    else:
        preprocessor = BreastCancerPP

    train_data, test_data  = preprocessor.process(dataset_file)
    
    # TODO: Update passed values 
    random_forest = RandomForest(arguments.number_of_trees)
    random_forest.train(train_data, max_depth, min_size, n_features)
    results = random_forest.predict(test_data)

    # TODO: Add code to check accuracy.


if __name__ == '__main__':
    main()
