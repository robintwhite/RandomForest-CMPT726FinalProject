FILES
=====
- BreastCancerDataSetPreprocessor.py - contains logic to take the breast cancer dataset and extract it into training and
                                       test sets.
- HockeyDataSetPreprocessor.py - contains logic to take the hockey dataset and extract it into training and test sets.
- RandomForest.py - contains logic for orchestrating learning and predicting with multiple trees.
- README.md - this read me file.
- runner.py - runs the random forests program.
- Tree.py - contains logic for a tree in the random forest.


System Requirements
===================
- must be run using Python 3.x
- numpy v 1.12.1
- pandas v 0.21.0


How to Run
==========
python runner.py -d <inputfile> --use_hockey_preprocessor


Output
======
```
Processing the data using Hockey Dataset Preprocessor...
I am tree number: 0
I am tree number: 1
I am tree number: 2
I am tree number: 3
Training on the given dataset!
Predicting on the given dataset!
```
