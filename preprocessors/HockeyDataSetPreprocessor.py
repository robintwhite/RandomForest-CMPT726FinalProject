def process(dataset_file):
   """
   Process the given hockey dataset so that it can be consumed by the Random Forest.

   @param dataset_file - filepath to hockey dataset.

   @return preprocessed data.

   """
   print("Processing the data using Hockey Dataset Preprocessor...")

   test_data_x = None
   test_data_y = None
   train_data_x = None
   train_data_y = None

   with open(dataset_file) as file_stream:
      None

   return train_data_x, train_data_y, test_data_x, test_data_y
