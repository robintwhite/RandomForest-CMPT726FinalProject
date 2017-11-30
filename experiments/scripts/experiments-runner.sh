#!/bin/bash
# Script to run experiments for various settings of runner.py.

# Experiment settings.
MAX_TREES=10
UPPER_DEPTH=10
MAX_FEATURES_PER_TREE=10

USE_GINI=0
USE_ENTROPY=0
USE_VARIANCE=0

VARY_NUMBER_OF_TREES=0
VARY_MAX_DEPTH=0
VARY_FEATURES_PER_TREE=0

# Experiment core settings.
# TODO: Adjust these so they are the same as Weka and Scikit-Learn.
NUMBER_OF_TREES=4
MAX_DEPTH=5
MIN_SPLIT_SIZE=2
NUMBER_OF_FEATURES=13


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
 -o  Output file name.
 -h  Print this usage message.

EOF
  exit
}


if [[ $# -eq 0 ]]
then
  usage
fi


# Parse out the settings for running the script.
while getopts ":i:o:gevtdfh" Option
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
    o)
      if [[ "${OPTARG}" != *.csv ]]
      then
        echo -e "\nA .csv file name must be passed to the -o option!\n"
        exit 1
      else
        OUTPUT_FILE=../results/${OPTARG}
        if [[ -f $OUTPUT_FILE ]]
        then
          RED="\033[0;31m"
          NO_COLOUR="\033[0m"
          echo -e "\n${RED}WARNING:${NO_COLOUR} The file $OPTARG already exists in the results directory!\n"
          read -p "Press [Enter] to append the results to the file or Ctrl+C to abort."
        fi
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

# Generate core command string.
BASE_COMMAND="python ../../runner.py -d $CSV_FILE --use_hockey_preprocessor"

if [[ $USE_GINI ]]
then
  BASE_COMMAND="$BASE_COMMAND --use_gini --target_label GP_greater_than_0"
elif [[ $USE_ENTROPY ]]
then
  BASE_COMMAND="$BASE_COMMAND --use_entropy --target_label GP_greater_than_0"
elif [[ $USE_VARIANCE ]]
then
  BASE_COMMAND="$BASE_COMMAND --use_variance --target_label sum_7yr_GP"
fi

echo -e "\nOkay, this part is probably going to take awhile so you will probably want to do something else in the" \
        "meantime.\n"
echo -e "TIP: Since each experiment uses a single process you might want to run the experiments in parallel.\n"

if [[ -n "$OUTPUT_FILE" ]]
then
  BASE_COMMAND="$BASE_COMMAND --output_file $OUTPUT_FILE"
  echo -e "Results will be written to $OUTPUT_FILE\n"
fi

if [[ $VARY_NUMBER_OF_TREES -eq 1 ]]
then
  echo -e "Running experiment to vary number of trees from 1 to $MAX_TREES...\n"
  for value in $(seq 1 $MAX_TREES)
  do
    COMMAND="$BASE_COMMAND --max_depth $MAX_DEPTH --min_split_size $MIN_SPLIT_SIZE --n_features $NUMBER_OF_FEATURES --number_of_trees $value"
    echo -e "Command: $COMMAND\n"
    $COMMAND
  done
elif [[ $VARY_MAX_DEPTH -eq 1 ]]
then
  echo -e "Running experiment to vary max depth from 1 to $UPPER_DEPTH...\n"
  for value in $(seq 1 $UPPER_DEPTH)
  do
    COMMAND="$BASE_COMMAND --max_depth $value --min_split_size $MIN_SPLIT_SIZE --n_features $NUMBER_OF_FEATURES --number_of_trees $NUMBER_OF_TREES"
    echo -e "Command: $COMMAND\n"
    $COMMAND
  done
elif [[ $VARY_FEATURES_PER_TREE -eq 1 ]]
then
  echo -e "Running experiment to vary number of features per tree from 1 to $MAX_FEATURES_PER_TREE...\n"
  for value in $(seq 1 $MAX_FEATURES_PER_TREE...)
  do
    COMMAND="$BASE_COMMAND --max_depth $MAX_DEPTH --min_split_size $MIN_SPLIT_SIZE --n_features $value --number_of_trees $NUMBER_OF_TREES"
    echo -e "Command: $COMMAND\n"
    $COMMAND
  done
fi