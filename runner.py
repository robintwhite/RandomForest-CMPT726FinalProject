"""
Script to help run the RandomForest program.

"""
from RandomForest import RandomForest
from argparse import ArgumentParser
from Sklearn_RF import Sklearn_RF
from datetime import datetime
from importlib import import_module

import csv
import os.path
import preprocessors.HockeyDataSetPreprocessor as HockeyPP



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
    mutually_exclusive_group.add_argument(
        '--use_variance',
        action='store_true',
        help="Use entropy for attribute splitting in the decision trees.")
    mutually_exclusive_group2 = argument_parser.add_mutually_exclusive_group()
    mutually_exclusive_group2.add_argument(
        '--use_hockey_preprocessor',
        action='store_true',
        help="Use hockey dataset preprocessing logic on the given dataset. (default)")
    mutually_exclusive_group2.add_argument(
        '--use_custom_preprocessor',
        help="Use custom dataset preprocessing logic on the given dataset.  Where USE_CUSTOM_PREPROCESSOR is the"
             "filename of the preprocessor file in the preprocessors directory to use, e.g. TemplateDataSetPreprocessor.")
    argument_parser.add_argument(
        '-d', '--data_file',
        required=True,
        help="File containing the dataset.")
    argument_parser.add_argument(
        '-t', '--number_of_trees',
        type=int,
        default=4,
        help="The number of trees to create for the random forest.")
    argument_parser.add_argument(
        '-m', '--max_depth',
        type=int,
        help="The maximum depth of all trees in the random forest.  (default: None).")
    argument_parser.add_argument(
        '-s', '--min_split_size',
        type=int,
        default=1,
        help="The threshold number of samples required at a node to stop further splitting.  (default: 1).")
    argument_parser.add_argument(
        '-f', '--n_features',
        type=int,
        help="The number of features to use when building each tree in the random forest.  Specifying None will use all"
              " the features (default: None).")
    argument_parser.add_argument(
        '-c', '--target_label',
        type=str,
        required=True,
        help="Target label that we want to predict.")
    argument_parser.add_argument(
        '-k','--sklearn_rf',
        action='store_true',
        help='Train and test dataset on SKlearn Random Forest')
    argument_parser.add_argument(
        '-w', '--number_of_workers',
        type=int,
        help="The number of workers to spawn during training of the random forest.  Specifying None will disable this"
             "feature. (default: None).")
    argument_parser.add_argument(
        '-o', '--output_file',
        help="Output file of the results.  If the file exists already, new entries will be appended to the end. (default: None).")
    argument_parser.add_argument(
        '-h', '--help',
        action='help',
        help="Show this message and exit.")
    arguments = argument_parser.parse_args()

    dataset_file = arguments.data_file
    output_file = arguments.output_file

    preprocessor = None

    if arguments.use_custom_preprocessor:
        preprocessor = import_module("preprocessors." + arguments.use_custom_preprocessor)
    else:
        preprocessor = HockeyPP

    #select class name
    class_name = arguments.target_label

    #select splitting cost function
    split_function = 'gini'

    if arguments.use_entropy:
        split_function = 'entropy'

    elif arguments.use_variance:
        split_function = 'variance'

    #Test regression with 'sum_7yr_GP'
    train_data, test_data  = preprocessor.process(dataset_file,class_name)

    random_forest = RandomForest(
        arguments.number_of_trees,
        arguments.max_depth,
        arguments.min_split_size,
        arguments.n_features,
        arguments.number_of_workers,
        split_function
    )

    t0 = datetime.now()
    random_forest.train(train_data, class_name)
    diff = datetime.now()-t0
    t = divmod(diff.days * 86400 + diff.seconds, 60)
    train_results = random_forest.bagging_predict(train_data)
    t0 = datetime.now()
    test_results = random_forest.bagging_predict(test_data)
    diff = datetime.now()-t0
    tp = divmod(diff.days * 86400 + diff.seconds, 60)

    if arguments.use_variance:
        train_accuracy = random_forest.mse(train_results, train_data[:,-1])
        print("\nTrain Mean squared error: {}".format(train_accuracy))

        test_accuracy = random_forest.mse(test_results, test_data[:,-1])
        print("Test Mean squared error: {}\n".format(test_accuracy))
    else:
        train_accuracy = random_forest.evaluate(train_results, train_data[:,-1])
        print("\nTrain Percent Correct: {}".format(train_accuracy))

        test_accuracy = random_forest.evaluate(test_results, test_data[:,-1])
        print("Test Percent Correct: {}\n".format(test_accuracy))

    print("\nTime for train: {}min {}sec".format(t[0],t[1]))
    print("Time for prediction: {}min {}sec\n".format(tp[0],tp[1]))

    if arguments.sklearn_rf is True:
        sk_rf = Sklearn_RF(
         arguments.number_of_trees,
         arguments.max_depth,
         arguments.min_split_size,
         arguments.n_features
        )

        # TODO: Need a sklearn_regression tree as well
        sk_rf.train(train_data,class_name)

        accuracy_sk = sk_rf.evaluate(test_data, tree_type = 'regressor' if arguments.use_variance else 'classifier')

        if arguments.use_variance:
            print('{}{}'.format('sklearn rf MSE: ',accuracy_sk))
        else:
            print('{}{}'.format('sklearn rf Percent correct: ',accuracy_sk*100))

    # Write out the results to a file, if one is specified, for downstream processing.
    if output_file:
        creating_new_file = True
        if os.path.isfile(output_file):
            creating_new_file = False

        headers = [
            "Features",
            "MaxDepth",
            "MinSplitThreshold",
            "Trees",
            "SplitCriteria",
            "Target",
            "TrainAccuracy",
            "TestAccuracy"
        ]

        with open(output_file, "a") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=headers, lineterminator="\n")
            if creating_new_file:
                print("Creating a new file {}!\n".format(output_file))
                writer.writeheader()
            else:
                print("Appending to the file {}!\n".format(output_file))
            writer.writerow({
                "Features" :  arguments.n_features if arguments.n_features else "ALL",
                "MaxDepth" : arguments.max_depth if arguments.max_depth else "NOLIMIT",
                "MinSplitThreshold" : arguments.min_split_size,
                "Trees" : arguments.number_of_trees,
                "SplitCriteria" : split_function,
                "Target" : class_name,
                "TrainAccuracy" : train_accuracy,
                "TestAccuracy" : test_accuracy
            })


if __name__ == '__main__':
    main()
