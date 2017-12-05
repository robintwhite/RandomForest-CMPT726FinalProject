def process(dataset_file, target_column):
   """
   Process the given dataset so that it can be consumed by the Random Forest.

   @param dataset_file - filepath to the dataset.
   @param target_column - the name of the column in the given dataset that we want to predict.

   @return preprocessed data where last column has target that we want to predict.

   """
   print("Processing the data using <Data Type> Dataset Preprocessor...")

   test_data = None
   train_data = None

   with open(dataset_file) as file_stream:
      None

   return train_data, test_data
