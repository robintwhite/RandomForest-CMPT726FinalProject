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
- sklearn v 0.19.1
- matplotlib v 2.1.0


How to Run
==========
python runner.py -d \<inputfile\> -c \<targetColumnName\> --use_hockey_preprocessor

Please use the -h/--help option for additional details.


Output
======
```
Processing the data using Hockey Dataset Preprocessor...
I am tree number: 0
I am tree number: 1
I am tree number: 2
I am tree number: 3
Predicting on the given dataset!
Predicting on the given dataset!

Train Percent Correct: 100.0
Test Percent Correct: 62.30366492146597


Time for train: 1min 4sec
Time for prediction: 0min 0sec
```


For Developers
==============
Please ensure that the pre-commit file in this repository is added as a Git hook for this project.

It will ensure that the runner.py script is run before making a commit, which will prevent people from breaking the this project.

To add the pre-commit hook, please do the following:
1. Ensure that you are in the mlclass-1777-randomforest directory.
2. Run: cp pre-commit .git/hooks -i

And you're done!


Compiling the report
==============
In case there is any trouble with the bib file, run pdfLatex + MakeIndex + BibTex to compile.
