#!/bin/bash
# Script to run experiments for various settings of runner.py.

# TODO: Define variables for settings for experiments.


function usage() {
cat <<EOF
Usage: `basename $0` -d DATA_FILE [-g|-e|-v] [-t|-d|-f][-h]

Script to run experiments for various settings of runner.py.

PARAMETERS:
 -i  Input file containing the dataset.
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
      fi;;
    g) echo "TODO";;
    e) echo "TODO";;
    v) echo "TODO";;
    t) echo "TODO";;
    d) echo "TODO";;
    f) echo "TODO";;
    h) usage;;
    :) echo "-$OPTARG requires an argument!";;
    *) echo "Unknown option $OPTARG given!"; exit 1;;
  esac
done

echo "TODO: COMPLETE THIS SCRIPT"