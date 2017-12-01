FILES
=====
experiments-runner.sh - script to automate the running of our experiments and generate results files.
README.md - this read me file.
sklearn_gridsearch_cv.py - script to run a gridsearch to find the optimal parameters of learning with the scikit-learn random forests.


System Requirements
===================
See README.md at root of project since this directory relies on the root one.


How to Run
==========
experiments-runner.sh

IMPORTANT: Be sure that you are in the directory mlclass-1777-randomforest/experiments/scripts before running the script.

bash experiments-runner.sh -i ../../preprocessors/hockeydataset.csv -e -t -o sample.csv

Please see the help message for details (-h option).


Output
======
```
Processing the data using Hockey Dataset Preprocessor...
Okay, this part is probably going to take awhile so you will probably want to do something else in the meantime.

TIP: Since each experiment uses a single process you might want to run the experiments in parallel.

Results will be written to ../results/test.csv

Running experiment to vary number of trees from 1 to 10...

Command: python ../../runner.py -d ../../preprocessors/hockeydataset.csv --use_hockey_preprocessor --use_gini --target_label GP_greater_than_0 --output_file ../results/test.csv --max_depth 5 --min_split_size 2 --n_features 13 --number_of_trees 1

Processing the data using Hockey Dataset Preprocessor...
I am tree number: 0
Predicting on the given dataset!
Predicting on the given dataset!

Train Percent Correct: 74.25431711145997
Test Percent Correct: 59.68586387434554

Creating a new file ../results/test.csv!


... more similar output ...

Command: python ../../runner.py -d ../../preprocessors/hockeydataset.csv --use_hockey_preprocessor --use_gini --target_label GP_greater_than_0 --output_file ../results/test.csv --max_depth 5 --min_split_size 2 --n_features 13 --number_of_trees 10

Processing the data using Hockey Dataset Preprocessor...
I am tree number: 0
I am tree number: 1
I am tree number: 2
I am tree number: 3
I am tree number: 4
I am tree number: 5
I am tree number: 6
I am tree number: 7
I am tree number: 8
I am tree number: 9
Predicting on the given dataset!
Predicting on the given dataset!

Train Percent Correct: 78.1789638932496
Test Percent Correct: 57.06806282722513

Appending to the file ../results/test.csv!
```