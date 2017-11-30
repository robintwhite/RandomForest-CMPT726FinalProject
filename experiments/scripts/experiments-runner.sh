#!/bin/bash
# Script to run experiments for various settings of runner.py.

# Experiment settings.
USE_GINI=0
USE_ENTROPY=0
USE_VARIANCE=0

VARY_NUMBER_OF_TREES=0
VARY_MAX_DEPTH=0
VARY_FEATURES_PER_TREE=0


function usage() {
cat <<EOF
Usage: `basename $0` -d DATA_FILE [-g|-e|-v] [-t|-d|-f][-h]

Script to run experiments for various settings of runner.py.

PARAMETERS:
 -g  Use the Gini index for attribute splitting in the
     decision trees.
 -e  Use entropy for attribute splitting in the decision
     trees.
 -v  Use variance for attribute splitting in the decision
     trees.

EXPERIMENTS:
 -t  Varying the number of trees in the random forest.
 -d  Varying the depth of trees in the random forest.
 -f  Varying the number of features each tree uses in the
     random forest.

OTHER:
 -i  Input file containing the dataset.
 -h  Print this usage message.

EOF
  exit
}


if [[ $# -eq 0 ]]
then
  usage
fi


# Parse out the settings for running the script.
while getopts ":i:gevtdfh" Option
do
  case $Option in
    i)
      if [[ "${OPTARG}" != *.csv ]]
      then
        echo -e "\nA .csv file must be passed to the -i option!\n"
        exit 1
      else
        CSV_FILE=${OPTARG}
      fi;;
    g) USE_GINI=1;;
    e) USE_ENTROPY=1;;
    v) USE_VARIANCE=1;;
    t) VARY_NUMBER_OF_TREES=1;;
    d) VARY_MAX_DEPTH=1;;
    f) VARY_FEATURES_PER_TREE=1;;
    h) usage;;
    :) echo "-$OPTARG requires an argument!";;
    *) echo "Unknown option $OPTARG given!"; exit 1;;
  esac
done

# Validate user input.
if [[ $((USE_GINI + USE_ENTROPY + USE_VARIANCE)) -gt 1 ]]
then
  echo "Only one of the following should be specified [gev]!"
  exit 1
elif [[ $((USE_GINI + USE_ENTROPY + USE_VARIANCE)) -eq 0 ]]
then
  echo "One of the following should be specified [gev]!"
  exit 1
elif [[ $((VARY_NUMBER_OF_TREES + VARY_MAX_DEPTH + VARY_FEATURES_PER_TREE)) -gt 1 ]]
then
  echo "Only one of the following should be specified [tdf]!"
  exit 1
elif [[ $((VARY_NUMBER_OF_TREES + VARY_MAX_DEPTH + VARY_FEATURES_PER_TREE)) -eq 0 ]]
then
  echo "One of the following should be specified [tdf]!"
  exit 1
elif [[ -z "$CSV_FILE" ]]
then
  echo "A CSV input file must be specified!"
  exit 1
elif [[ ! -f $CSV_FILE ]]
then
  echo "The provided file $CSV_FILE doesn't exist or isn't a file!"
  exit 1
fi