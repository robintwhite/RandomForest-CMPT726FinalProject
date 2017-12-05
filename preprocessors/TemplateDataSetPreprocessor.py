def process(dataset_file):
   """
   Process the given breast cancer dataset so that it can be consumed by the Random Forest.

   @param dataset_file - filepath to breast cancer dataset.

   TODO: Rename this file if we decide to use a different dataset.

   @return preprocessed data.

   """
   print("Processing the data using Breast Cancer Dataset Preprocessor...")

   test_data_x = None
   test_data_y = None
   train_data_x = None
   train_data_y = None

   with open(dataset_file) as file_stream:
      None

   return train_data_x, train_data_y, test_data_x, test_data_y
