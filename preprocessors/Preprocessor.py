"""
Script to preprocess the data in the given file.


"""

import pandas as pd

COLUMNS_TO_USE_GP_greater_than_0 = [
    "DraftAge",
    "country_group",
    "Height",
    "Weight",
    "Position",
    "DraftYear", # Remove at a later step.
    "CSS_rank",
    "rs_GP",
    "rs_G",
    "rs_A",
    "rs_P",
    "rs_PIM",
    "rs_PlusMinus",
    "po_GP",
    "po_G",
    "po_A",
    "po_P",
    "po_PIM",
    "GP_greater_than_0"
]

COLUMNS_TO_USE_sum_7yr_GP = [
    "DraftAge",
    "country_group",
    "Height",
    "Weight",
    "Position",
    "DraftYear", # Remove at a later step.
    "CSS_rank",
    "rs_GP",
    "rs_G",
    "rs_A",
    "rs_P",
    "rs_PIM",
    "rs_PlusMinus",
    "po_GP",
    "po_G",
    "po_A",
    "po_P",
    "po_PIM",
    "sum_7yr_GP"
]


def createDummies(all_data, discrete_columns):
    all_data_with_dummies = all_data
    for discrete_column in discrete_columns:
        one_hot_encoding = pd.get_dummies(all_data_with_dummies[discrete_column])
        all_data_with_dummies = all_data_with_dummies.drop(columns=[discrete_column])
        all_data_with_dummies = all_data_with_dummies.join(one_hot_encoding)

    return all_data_with_dummies


def standardizeData(all_data_with_dummies, columns_to_ignore):
    columns = set(all_data_with_dummies.columns) - columns_to_ignore
    standardized_data = all_data_with_dummies

    for column in columns:
        std = standardized_data[column].std()
        mean = standardized_data[column].mean()
        standardized_data[column] = standardized_data[column] - mean
        standardized_data[column] = standardized_data[column] / std

    return standardized_data


def addBiasColumn(all_data):
    number_of_rows = len(all_data)

    return all_data.assign(bias = [1] * number_of_rows)


def splitIntoTrainingAndTest(all_data, split_column, train_values, test_value, target_column, shuffle):
    """
    Split the data into training and test sets and then into feature and target groups.

    @param all_data - the preprocessed dataset to split.
    @param split_column - the column used to determine the split between training and test sets.
    @param train_values - rows containing any of these values for their split_column will be included for the training
                          set.
    @param test_value - rows containing this value for their split_column will be included for the test set.
    @param target_column - the column to use for the target.
    @param shuffle - True if training set should be shuffled; otherwise False.

    """
    # Separate into test and training sets.
    train_data = all_data[all_data[split_column].isin(train_values)]
    test_data = all_data[all_data[split_column] == test_value]

    # Remove split_column column.
    train_data = train_data.drop(columns=[split_column])
    test_data = test_data.drop(columns=[split_column])

    # Randomly shuffle the rows to prevent any patterns in the distribution of the dataset from affecting the results.
    if shuffle:
        train_data = train_data.sample(frac=1, random_state=1)

    # Separate into feature and target sets.
    x_train = train_data.ix[:, train_data.columns != target_column]
    y_train = train_data[target_column]
    x_test = test_data.ix[:,train_data.columns != target_column]
    y_test = test_data[target_column]

    return x_train, y_train, x_test, y_test


def process(dataset_file_path , split_column, train_values, test_value, target_column, shuffle=False):
    """
    Preprocess the given file and return the following 4 data groupings:
    1) Training features data.
    2) Training target data.
    3) Test features data.
    4) Test target data.

    @param split_column - the column used to determine the split between training and test sets.
    @param train_values - rows containing any of these values for their split_column will be included for the training
                          set.
    @param test_value - rows containing this value for their split_column will be included for the test set.
    @param target_column - the column to use for the target.
    @param shuffle - True if training set should be shuffled; otherwise False.

    @return 4 data groupings for training and validating supervised machine learning algorithms.

    """

    all_data = None
    # Load data.
    if target_column == "sum_7yr_GP":
        all_data = pd.read_table(dataset_file_path, delimiter=",", usecols=COLUMNS_TO_USE_sum_7yr_GP)
    else:
        all_data = pd.read_table(dataset_file_path, delimiter=",", usecols=COLUMNS_TO_USE_GP_greater_than_0)

    # Preprocess the data.
    all_data_with_dummies = createDummies(all_data, ["country_group", "Position"])
    columns_to_ignore = set(all_data_with_dummies.columns).difference(set(all_data.columns))
    columns_to_ignore.add(target_column)
    columns_to_ignore.add(split_column)
    standardized_data = standardizeData(all_data_with_dummies, columns_to_ignore)

    return splitIntoTrainingAndTest(standardized_data, split_column, train_values, test_value, target_column, shuffle)
