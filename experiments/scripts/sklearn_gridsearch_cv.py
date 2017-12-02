
# coding: utf-8

# In[ ]:

from Sklearn_RF import Sklearn_RF
import preprocessors.HockeyDataSetPreprocessor as HockeyPP
import preprocessors.BreastCancerDataSetPreprocessor as BreastCancerPP
import preprocessors.Preprocessor as pp
from argparse import ArgumentParser

def main():
    """
    run cross-validation grid search on a sklearn RandomForestClassifier and RandomForestRegressor for the hockey dataset (might have to set aside another option for the breast cancer dataset.)

    """

    argument_parser = ArgumentParser(
        description="Script to run the RandomForest program.",
        add_help=False)

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

    arguments = argument_parser.parse_args()

    sk_rf = Sklearn_RF(
         arguments.number_of_trees,
         arguments.max_depth,
         arguments.min_split_size,
         arguments.n_features
       )

    preprocessor = HockeyPP

    train_data, test_data  = preprocessor.process('../../preprocessors/hockeydataset.csv','GP_greater_than_0')

    accuracy_sk = sk_rf.gridSearch(train_data,test_data,'classifier')

    train_data, test_data  = preprocessor.process('../../preprocessors/hockeydataset.csv','sum_7yr_GP')

    accuracy_sk = sk_rf.gridSearch(train_data,test_data,'regressor')

if __name__ == '__main__':
    main()
