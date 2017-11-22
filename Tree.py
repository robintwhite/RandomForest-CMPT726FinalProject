from random import seed
from random import randrange
from csv import reader
from math import sqrt
import pandas as pd
import numpy as np

class Tree():
   """
   Class to hold logic for a tree in a random forest.

   """

    def __init__(self, tree_id,dataset,max_depth,min_size,n_features):
       """
       Initialize resources for a tree.

       @param tree_id - TODO: Replace parameters with what we need.

       """
       # TODO: Add code here as necessary.
       self.id = tree_id
       self.dataset = dataset
       self.max_depth = max_depth
       self.min_size = min_size
       self.n_features = n_features

    def printID(self):
       """
       TODO: Remove this method and add methods needed for a tree in a random forest.

       """
       print("I am tree number: {}".format(self.id))

    #optimized test_split function using numpy array. Split dataset into groups less than or greater than
    #an attribut value.
    #Created by: Jae Lee
    def test_split(index,value,dataset_t):

        left =dataset_t[dataset_t[:,index]<value]

        right =dataset_t[dataset_t[:,index]>=value]

        return left, right

    #Calculate gini_index score for groups split based on whether sample's specific attribute
    #value is greater than or equal to a chosen split point value
    #Created by: Jae Lee
    def gini_index(groups):

        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))

        # sum weighted Gini index for each group
        gini = 0.0

        for group in groups: 
            # score the group based on the score for each class
            score_t = gini_index_grp_score(group)

            gini += (1.0 - score_t) * (float(len(group)) / n_instances)

        return gini

    #Calculate group score by split - apply- combine samples by labels and counting number of each labels.
    #Score is the sum of each count divided by the total size of the group squared. 
    #Created by: Jae Lee
    def gini_index_grp_score(group_t):

        size = float(len(group_t))
            # avoid divide by zero
        if size == 0:
            return 0.0

        #count number of unique labels (column -1) and return total number of counts
        labels, counts = np.unique(group_t[:,-1],return_counts=True)

        #score is equal to sum((each_label_count/size_group)^2)
        score_t = ((counts/size)**2).sum()

        return score_t

    #get gini_score for dataset split on each row's indexed attribute value
    #created by: Jae Lee
    def get_row_score(row,index,dataset):

        groups = test_split(index, row[index], dataset)

        gini = gini_index(groups)

        return gini

    # Select the best split point for a dataset. This routine and related subroutines convert the dataset
    #into a 2D numpy array. Rest of the functions are still using dataset in a python list format.
    #So extra computation time is taken at the end of the function to convert dataset back into a python list.
    #I think switching rest of the functions to use dataset in the numpy array format will improve runtime.
    #created by: Jae Lee
    def get_split(dataset, n_features):

        b_index, b_value, b_score, b_groups = 999, 999, 999, None

        count_all_features = len(dataset[0])-1

        #randomly select number of features. More concise method of generating random int array.
        #But I have commented this out for now since we need to obtain same results as the tutorial for 
        #testing purposes
        #features = np.random.randint(count_all_features,size=n_features)        

        features = list()
        while len(features) < n_features:
            index = randrange(len(dataset[0])-1)
            if index not in features:
                features.append(index)

        dataset_t = np.array(dataset)

        for index in features:

            #returns all gini index scores of each row for the selected feature
            scores = np.apply_along_axis(get_row_score,1,dataset_t,index,dataset_t)

            current_b_score = np.min(scores)

            current_b_row = np.argmin(scores)

            groups = test_split(index, dataset_t[current_b_row,index], dataset_t)

            current_value = dataset_t[current_b_row,index]

            if current_b_score < b_score:
                groups_t = []

                #have to convert np array back to python list since rest of the functions take data as a list
                #If rest of the functions are converted to use the data in np array format, I can get rid of this
                for g in groups:
                    group = g.tolist()
                    groups_t.append(group)

                groups_t = tuple(groups_t)

                b_index, b_value, b_score, b_groups = index, current_value, current_b_score, groups_t

        return {'index':b_index, 'value':b_value, 'groups':b_groups}

    #util function to split child nodes starting from root
    def tree_build_util(root):
    
    