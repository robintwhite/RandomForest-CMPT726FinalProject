"""
Script to help run the RandomForest program.

"""
from RandomForest import RandomForest
from argparse import ArgumentParser
from Sklearn_RF import Sklearn_RF
import preprocessors.HockeyDataSetPreprocessor as HockeyPP
import preprocessors.BreastCancerDataSetPreprocessor as BreastCancerPP
import preprocessors.Preprocessor as pp



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
        '-k','--sklearn_rf',
        action='store_true',
        help='Train and test dataset on SKlearn Random Forest')
    argument_parser.add_argument(
        '-w', '--number_of_workers',
        type=int,
        help="The number of workers to spawn during training of the random forest.  Specifying None will disable this"
             "feature. (default: None).")
    argument_parser.add_argument(
        '-c', '--target_label',
        type=str,
        required=True,
        help="Target label that we want to predict.")
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
    
    #select class name
    class_name = arguments.target_label
    
    #select splitting cost function
    split_function = 'gini'
    
    if arguments.use_entropy is True:
        split_function = 'entropy'
    
    elif arguments.use_variance is True:
        split_function = 'variance'
        
    #Test regression with 'sum_7yr_GP'
    train_data, test_data  = preprocessor.process(dataset_file,class_name)
    #train_x, train_y, test_x, test_y = pp.process(dataset_file, "DraftYear", [2004, 2005, 2006], 2007, "GP_greater_than_0")

    random_forest = RandomForest(
        arguments.number_of_trees,
        arguments.max_depth,
        arguments.min_split_size,
        arguments.n_features,
        arguments.number_of_workers
    )
    random_forest.train(train_data, class_name,split_function)
    results = random_forest.bagging_predict(test_data)
    
    # TODO: Add code to check accuracy. 
    # TODO: Need separate code for checking regression accuracy
    accuracy = random_forest.evaluate(results, test_data[:,-1])
    print('{}{}'.format("Percent correct: ", accuracy))

    if arguments.sklearn_rf is True:
        sk_rf = Sklearn_RF(
         arguments.number_of_trees,
         arguments.max_depth,
         arguments.min_split_size,
         arguments.n_features
        )
        
        sk_rf.train(train_data,class_name)
        
        accuracy_sk = sk_rf.evaluate(test_data)

        print('{}{}'.format('sklearn rf Percent correct: ',accuracy_sk*100))

if __name__ == '__main__':
    main()