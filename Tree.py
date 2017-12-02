import numpy as np
import math

class Tree():
    """
    Class to hold logic for a tree in a random forest.

    """
    
    # TODO: Maybe we need a pruning method?

    def __init__(self, tree_id, max_depth, min_split_size, n_features):
        """
        Initialize resources for a tree.

        @param tree_id - id of tree.
        @param max_depth - maximum depth of the tree.
        @param min_split_size - the minimum number of samples required at a node to allow for further splitting.
        @param n_features - the number of features to be used when building each tree.

        """
        self.id = tree_id
        self.max_depth = max_depth
        self.min_split_size = min_split_size
        self.n_features = n_features
        self.root = None

    def printID(self):
        """
        TODO: Remove this method and add methods needed for a tree in a random forest.

        """
        print("I am tree number: {}".format(self.id))
    
    def var_index(self,groups):

        tot_var = 0
        
        for g in groups:

            v = np.var(g)
            
            if not g.any():
                
                continue
            
            tot_var += v
        
        return tot_var
    
    def entropy_index(self, groups,num_labels):
        """
          Calculate information gain with entropy
        """
        n_instances = float(sum([len(group) for group in groups]))

        # sum weighted Gini index for each group
        entropy = 0.0

        for group in groups:
            # score the group based on the score for each class
            score_t = self.entropy_grp_score(group)

            entropy += (score_t) * (float(len(group)) / n_instances)

        return entropy

    def entropy_grp_score(self, group_t):

        size = float(len(group_t))
            # avoid divide by zero
        if size == 0:
            return 0.0

        #count number of unique labels (column -1) and return total number of counts

        labels, counts = np.unique(group_t,return_counts=True)

        #score is equal to sum((each_label_count/size_group)^2)
        
        score_t = (np.log2(counts/size)*(counts/size)).sum()*-1

        return score_t

    def gini_index(self, groups, num_labels):
        """
        Calculate gini_index score for groups split based on whether sample's specific attribute
        value is greater than or equal to a chosen split point value
        """
        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))

        # sum weighted Gini index for each group
        gini = 0.0

        for group in groups:
            # score the group based on the score for each class
            score_t = self.gini_index_grp_score(group,num_labels)

            gini += (1.0 - score_t) * (float(len(group)) / n_instances)

        return gini

    def gini_index_grp_score(self, group_t, num_labels):

        size = float(len(group_t))
            # avoid divide by zero
        if size == 0:
            return 0.0

        #count number of unique labels (column -1) and return total number of counts

        labels, counts = np.unique(group_t,return_counts=True)

        #score is equal to sum((each_label_count/size_group)^2)
        score_t = ((counts/size)**2).sum()

        return score_t

    def test_split_justlabels(self,index,value,dataset_t):
        """
        Split dataset into groups less than or greater than
        an attribute value.
        """
        left =dataset_t[dataset_t[:,index]<value][:,-1]

        right =dataset_t[dataset_t[:,index]>=value][:,-1]

        return left, right

        
    #optimized test_split function using numpy array.
    def test_split(self, index, value, dataset_t):
        """
        Split dataset into groups less than or greater than
        an attribute value.
        """
        left =dataset_t[dataset_t[:,index]<value]

        right =dataset_t[dataset_t[:,index]>=value]
     

        return left, right

    def entropy_row_score(self, row, index, dataset, parent_entropy, num_labels):

        """
        get entropy_score for dataset split on each row's indexed attribute value
        """

        groups = self.test_split_justlabels(index, row[index], dataset)

        entropy_score = parent_entropy + self.entropy_index(groups,num_labels)

        return entropy_score


    def gini_row_score(self, row, index, dataset, num_labels):
        """
        get gini_score for dataset split on each row's indexed attribute value
        """
        groups = self.test_split_justlabels(index, row[index], dataset)

        gini = self.gini_index(groups,num_labels)

        return gini

    def variance_row_score(self,row,index,dataset):
        
        groups = self.test_split_justlabels(index, row[index], dataset)
        
        variance = self.var_index(groups)
        
        return variance
        
    def get_split(self, dataset, n_features, splitf):
        """
        Select the best split point for a dataset.
        This routine and related subroutines convert the dataset
        into a 2D numpy array. Rest of the functions are still using dataset in a python list format.
        So extra computation time is taken at the end of the function to convert dataset back into a python list.
        I think switching rest of the functions to use dataset in the numpy array format will improve runtime.
        """
        b_index, b_value, b_score, b_groups = math.inf, math.inf, math.inf, None
        dataset_t = np.array(dataset)
        count_all_features = len(dataset[0])-1
        if not n_features:
            n_features = count_all_features

        #count number of unique labels (column -1) and return total number of counts
        labels, num_labels = np.unique(dataset_t[:,-1], return_counts=True)

        #randomly select number of features.
        #seed for testing
        #np.random.seed(7)
        features = np.random.randint(count_all_features,size=n_features)

        for index in features:

            scores = None
            #returns all gini index scores of each row for the selected feature
            if splitf == 'gini':

                scores = np.apply_along_axis(self.gini_row_score,1,dataset_t,index,dataset_t,num_labels)
            #returns all entropy scores of each row for the selected feature
            elif splitf == 'entropy':

                parent_entropy = self.entropy_index(dataset_t,num_labels)

                scores = np.apply_along_axis(self.entropy_row_score,1,dataset_t,index,dataset_t,parent_entropy, num_labels)

            #variance split function for regression tree
            elif splitf == 'variance':
                
                scores = np.apply_along_axis(self.variance_row_score,1,dataset_t,index,dataset_t)
                
                #print(scores)
                
            current_b_score = np.min(scores)

            current_b_row = np.argmin(scores)

            current_value = dataset_t[current_b_row,index]

            if current_b_score < b_score:
                groups = self.test_split(index, dataset_t[current_b_row,index], dataset_t)

                groups_t = tuple(groups)

                b_index, b_value, b_score, b_groups = index, current_value, current_b_score, groups_t

        return {'index':b_index, 'value':b_value, 'groups':b_groups}


    # Create a terminal node value
    def to_terminal(self, group,splitf):
        
        if not group.any():
            return 0.
        
        outcomes = [row[-1] for row in group]
        
        if splitf =='variance':
            mean = np.round(np.mean(outcomes),3)
            
            if np.isnan(mean):
                mean = 0.
                
            return mean
        
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count) #majority vote

    def split(self, node, max_depth, min_size, n_features, depth,splitf):
        """
        split child nodes starting from root
        """
        #print("depth: ",depth)
        left, right = node['groups']
        del(node['groups'])
        
         # check for a no split
        if not left.any() or not right.any():
            arr = np.vstack((left,right))
            node['left'] = node['right'] = self.to_terminal(arr,splitf)
            return
       
        # check for max depth
        if max_depth is not None and depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left,splitf), self.to_terminal(right,splitf)
            return
        # process left child
        if len(left) <= min_size:
            node['left'] = self.to_terminal(left,splitf)
        else:
            node['left'] = self.get_split(left, n_features,splitf)
            self.split(node['left'], max_depth, min_size, n_features, depth+1,splitf)
        # process right child
        if len(right) <= min_size:
            node['right'] = self.to_terminal(right,splitf)
        else:
            node['right'] = self.get_split(right, n_features,splitf)
            self.split(node['right'], max_depth, min_size, n_features, depth+1,splitf)

    def tree_build_util(self, train_data, target_class,splitf):
        """
        util function to build tree
        @param train_data - training dataset
        @param target_class - the column we want to be able to predict

        Note: This method is basically the fit() method.

        """
        split_point = self.get_split(train_data, self.n_features,splitf)
#         
        self.split(split_point, self.max_depth, self.min_split_size, self.n_features, 1,splitf)
        self.root = split_point

    def predict(self, node, row):
        # tree structure is dictionary of index (feature that is split), value (value for left and right)
        # and output if terminal leaf for left and right or another dictionary of next level
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict): #is it not a termanl node
                return self.predict(node['left'], row) #run on this node, working way down tree
            else:
                return node['left'] #if terminal, return value either 1 or 0
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right'] #if terminal, return value either 1 or 0