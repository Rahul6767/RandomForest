# -*- coding: utf-8 -*-
"""ID3withMax_value_error.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1derTzP0gT9ZZSrbH7rHTwZIzhlirZw9m
"""

# -*- coding: utf-8 -*-
"""ID3withEntropy.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1derTzP0gT9ZZSrbH7rHTwZIzhlirZw9m
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 12:11:08 2024

@author: shahriar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import os
import time


import sys

# Arithmetic
from fractions import Fraction
from decimal import Decimal

from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')



from scipy.stats import chi2_contingency


import math

class ID3:

    def __init__(self, discrete_features=[], continuous_features=[], do_chi_test=False, confidence_level=0.90):
        self.counter = 0
        self.tree = []
        #self.labels = []
        self.leaf_nodes = []
        self.attributes = []

        self.confidence_level = confidence_level

        self.discrete_features = discrete_features
        self.continuous_features = continuous_features
        self.do_chi_test = do_chi_test

        #self.discretize_continuous_attributes()


    def chi_squared_test(self, actual_table, confidence_level=0.90):
        """
        INPUT
        ----

        contingency_table
        ---
            pandas df - The contigency_table is has value count from examples_vi - the actual one


        RETURNS
        ----

        Boolean variable : True or False

        """

        chi2, pval, dof, expected_freqs = chi2_contingency(actual_table)

        alpha = 1 - confidence_level

        if pval < alpha:
            print(f"Feature has a significant association with the target variable (p-value: {pval:.4f}).")
            return True


        else:
            print(f"Feature does not have a significant association with the target variable (p-value: {pval:.4f}).")
            return False


    def print_tree_list(self):
        print(self.tree)


    def is_trained(self):

        if len(self.tree) > 0:
            return True

        return False

    def is_leaf_node(self, node, leaf_nodes):

        for leaf in leaf_nodes:
            if node in leaf:
                return True, leaf[node]

        return False, None



    def list_attribute_names(self):

        if self.is_trained() == False:

            print("Error: tree is not trained.")
            return None

            # get the names of all attributes

        for a in self.tree:

            print(a[0])
            #first entry of array a is the attribute name
            self.attributes.append(a[0])


        print('attributes : ', self.attributes)


    def predict_tuple(self, tuple_):

        """

        INPUT is a list of (attribute, value) tuples

        -Sample input : [('outlook', 'sunny'), ('temp', 'hot'), ('humidity', 'high'), ('windy', 'False'), ('play', 'no')]


        RETURNS
        --------
            the label according to the trained tree

        """

        prev_nodes_ = []

        print('root in tree : ', self.tree[-1][0])

        for attr_val in reversed(self.tree) :

            """
                Inside the tree. tree[0][0] is the root.
            """

            print('--- reversed tree propagation : ', attr_val)
            print('--- prev_nodes_ : ', prev_nodes_)
            # check if attr_val[0] is in one of the tuples , if it is present, store it in a variable named attr
            tree_0 = attr_val[0]


            # in the tuple_ list, get the value for the corresponding value for the respective tree_attr

            for _ in tuple_:
                """
                    _ denotes a single entry in the tuple
                """

                if _[0] == tree_0 :

                    #test_val = _[1]

                    print('---------')

                    print('tree_0 : ', tree_0)
                    print('_[0] : ', _[0])

                    print('=========')

                    """
                     -Keep checking in test_attr.
                     -Now, iterate across each attr_val list
                    """

                    for node_val in attr_val[1: ]:
                        print('\t\t node val : ', node_val)


                        # take the value for that attribute
                        value_  = node_val.split('->')[1]
                        print('value from .split : ', value_)
                        print("\t\t_[1] :" , _[1])

                        #print('\t\t_[1] :', _[1])

                        """
                        after split, check value is the second entry of tuple _
                        """


                        if _[1] == value_ :

                            print('\t\t\t-------------in the base case------------------')
                            if len(prev_nodes_) == 0:
                                print('\t\t\t _[1] and value_ are the same')
                                print('\t\t\t _[1] : ', _[1], ', value_ : ', value_)
                                print('\t\t\t _[0] : ', _[0] )

                                prev_nodes_.append( tree_0 +'-' +_[1] )

                                print('\t\t\t prev_node : ', prev_nodes_)

                                """
                                Now check if the tuple-entry is a leaf node

                                """


                                leaf_node_ , label_ = self.is_leaf_node( _[0]+'-'+_[1], self.leaf_nodes)

                                print( "\t\t\tleaf_node_ : ", leaf_node_ , ' label_ : ', label_)


                                if label_ == None:
                                  return 1

                                elif leaf_node_ :
                                    print('\t\t\tlabel for this tuple : ', label_)
                                    return label_


                                #if len(prev_nodes_ ) != 0:
                                #    print('\t\t\tthere is an entry is prev node')
                                #    continue

                            else: # after the root is discovered and onwards
                                print('\t\t\t------------in the induction case------------')

                                print('\t\t\t _[1] and value_ are the same')
                                print('\t\t\t _[1] : ', _[1], ', value_ : ', value_)
                                print('\t\t\t _[0] : ', _[0] )


                                prev_node_ = prev_nodes_[-1]

                                print('\t\t\t prev_node_ : ', prev_node_)
                                print('\t\t\t node_val : ', node_val)




                                if prev_node_ in node_val:
                                    print('\t\t\t-------------------prev_node_ in node_val')

                                    # split on '<-'
                                    pv_, next_node_ = node_val.split('<-')
                                    del pv_

                                    print('\t\t\t-------------------next_node : ', next_node_)

                                    # split next_node

                                    attr_, val_ = next_node_.split('->')

                                    print('\t\t\t-------------------attr_ : ', attr_)
                                    print('\t\t\t-------------------val_ in tree node: ', val_)
                                    print('\t\t\t-------------------value_ in tuple: ', value_)

                                    prev_nodes_.append(node_val)

                                    leaf_node_ , label_ = self.is_leaf_node( _[0]+'-'+_[1], self.leaf_nodes)

                                    print( "\t\t\tleaf_node_ : ", leaf_node_ , ' label_ : ', label_)

                                    if label_ == None:
                                      return 1

                                    elif leaf_node_ :
                                        print('\t\t\tlabel for this tuple : ', label_)
                                        return label_
                                        print('\t\t\t------------end of induction case------------')
                                    print('\t\t\t------------end of induction case------------')





                                    """
                                    value_ :
                                    val_ :
                                    """
                                    # check if this attr_ val_ pair is in the tuple


                            print('\t\t\t-------------------------------')
                    #sys.exit(0)
            print('---- prev_nodes_ : ', prev_nodes_)
            print('end of function')


                    #if(test_val)
            #sys.exit(0)


    def predict(self, test_data):

        X_pred = []

        # Iterate over each tuple in test_data
        for index, row in test_data.iterrows():

            tuple_ = []

            #print('index : ', index )
            #print('row :\n', row)
            #print('\n')
            for col_name, value in row.items():

                #print(f"\t\tColumn Name: {col_name}, Value: {value}")



                tuple_.append( (col_name, value) )

                #results.append(  predict_tuple( (col_name, value) )   )

                #results.append( predict_tuple(tuple_) )


            #print('data as as list of row, col tuple')
            print(tuple_)

            X_pred.append( self.predict_tuple(tuple_) )

            #print('*** x_pred = ', x_pred)
        return np.array( X_pred )



    def run(self, examples, target_attribute, attributes, prev_node=None):

        """

        Parameters
        ----------
        examples : training examples
            pandas df

        target_attribute : attriburte whose value is to be predicted by tree
            pandas series

        attributes : list of other attributes to be tested
            list w/ strings

        Returns
        -------
        Returns a decision tree that correctly classsifies the examples (!)



        AMENDMENT: The input parameter should look as follows:
            examples, target_attribute, attributes
        """


        print('-----------------------------')
        print('-----run method called -----')

        #print( 'examples.head() : \n', examples.head() )
        #print('examples.shape : ', examples.shape)

        #print( 'target_attribute.head() :\n', target_attribute.head() )
        #print( 'target_attribute.shape:\n', target_attribute.shape )

        print('attributes :\n', attributes)


        #examples = df.iloc[:, 0:-1]
        #target_attribute = df.iloc[:, -1]
        #attributes = attributes

        # 1. create  a root for the tree
        root = []
        #labels = []

        # 2 and 3. All examples have same label -> return single node with that label
        target_values, target_counts = np.unique(target_attribute, return_counts=True)

        if len(target_values) ==1:

            print('**** CONDITION 2 & 3 ****')
            print('---> returning, root = ', root, ', target_values[0] = ', target_values[0], ', for prev_node = ', prev_node)
            #print('---> type of target_values[0]', type(target_values[0])
            self.leaf_nodes.append( {prev_node: list(target_values)[0] })
            #labels.append( target_values[0])

            return root, target_values[0] # I am not sure how this will look

        # 4.attributes is empty
            # then return single-node tree root,
            # with most common label in target_attribute in examples

        if len(attributes) == 0:

            # Find the most frequent target value
            target_value, target_count = np.unique(target_attribute, return_counts=True)
            most_frequent_value = target_value[np.argmax(target_count)]

            print('**** CONDITION 4 ****')
            print('returning, root = ', root ,
                  ', most_frequent_value =', most_frequent_value ,
                  'for prev_node = ', prev_node)


            print('type of most_frequent_value : ', type(most_frequent_value) )

            self.leaf_nodes.append( {prev_node: most_frequent_value})
            #labels.append(most_frequent_value)

            #sys.exit(0)

            return root, most_frequent_value

        # 5. Do this - this is where is information gain is calculated
        """
        Find the information_gain for each attribute to decide the best att

            - A is the variable with the best attribute that best classifies the examples
            - examples_vi
            -
        """

        A = None # This is th attribute with the highest info gain

        highest_info_gain = float('-inf')

        for a in attributes: # find the attribute with the highest information gain

            """
            There needs to be a methods to decide if the attribute is discrete or continuous.
            For now, assume all are discrete.
            """

            #info_gain = self.information_gain( examples[a], target_attribute, 'entropy') # examples is pd df

            #print('----------------------------')
            #print('\t\t\tDEBUGGING KEY ERROR in info gain for disc attr')

            #display('\t\texamples[a]', examples[a])
            #display('\t\ttarget_attribute ', target_attribute)

            #print('----------------------------')


            info_gain = self.information_gain_for_discrete_attribute(examples[a],
                                                                     target_attribute,
                                                                     impurity = 'max_value_error')

            if info_gain > highest_info_gain:
                A = a
                highest_info_gain = info_gain

        root.append(A)

        print('root : ', root)
        print('highest_info_gain : ', highest_info_gain)
        #sys.exit()


        # get the unique values in A - all the possible values the attribute may have


        vi_list_np = np.unique(examples[A]) # examples is pandas df
        #vi_with_root = []
        print('----- vi_list_np is the unique values that the attr w/ highest info gain has -----')

        print('vi_list_np : ', vi_list_np)
        #print('vi_with_root : ', vi_with_root)

        print('------------------------------')

        vi_count = 0



        print('---- As of now, I know the attribute A : ', A, ' that has the highest information gain ----')
        print('---- As of now, I know the attr_vals of A : ', vi_list_np , ' ----')
        print('---- I need these A vi_list_np to start w/ chi-squared test', ' ----')





        print('--------CHI-SQUARE TEST ---------')

        actual_table = pd.crosstab(examples[A], target_attribute)

        chi_sq_test = self.chi_squared_test(actual_table)
        #display('actual table\n\n', actual_table)

        print('chi-sq_test result for attribute ', A, ' : ', chi_sq_test)


        if self.do_chi_test :


            if chi_sq_test :
                print('--- CHI SQUARE TEST IS PASSED---')

            else:
                print('--- CHI SQUARE TEST IS NOT PASSED---')

                # Find the most frequent target value
                target_value, target_count = np.unique(target_attribute, return_counts=True)
                most_frequent_value = target_value[np.argmax(target_count)]

                print('**** CHI-SQUARE TEST not passes and we do not increase the tree any further for attr ', A ,'****')
                print('returning, root = ', root ,
                      ', most_frequent_value =', most_frequent_value ,
                      'for prev_node = ', prev_node)


                #print('type of most_frequent_value : ', type(most_frequent_value) )

                if prev_node == None:

                    self.tree.append(root[-1])

                    for vi in vi_list_np:

                        self.tree.append(str(A) + '->' + str(vi))


                        key = str(A) + '-' + str(vi)
                        self.leaf_nodes.append( {key: most_frequent_value })
                        #labels.append(most_frequent_value)


                    """
                    If tree has no node, return most_frequent value. Else, stop expanding tree.
                    """


                    print('--->>> self.tree value, before returning : ', self.tree)
                    print('--->>> type(root[-1]) : ', type(root[-1]))


                return root[-1], most_frequent_value # got this out of if statement

                #return


        print('-----------------')








        """
        At this point, I have successfully conducted chi_sq test. I need to use this

        """

        # At this point, I have the actual table. I need the expected table.

        """
        Realization : tecnically, I DON'T need the expected table. The library/function can do this
        """


        #sys.exit(0)


        for vi in vi_list_np :

            vi_count = vi_count+1 # where/why am I using this?

            # Add a new branch below root, corresponding to the test A = v_i
            #vi_with_root.append(A+'->'+vi) # a list
            #root.append( vi_with_root )


            next_ = None

            if prev_node is None:

                root.append(A+'->'+str(vi))
            else :
                root.append(prev_node +'<-' + A + '->'+str(vi))

            """
            See how root looks.

            Even with the chi_sq test, the attr_vals are

            """

            #print('root after adding attr values : ', root)

            #print('----\n\n')

            examples_vi  = examples[ examples[A] == vi ].drop(columns=A) # works


            print('-----------------')
            print('---- examples_vi df is the subset df of A; printed below: ')
            print( examples_vi )
            #print('\t\texamples_vi df')
            print('-----------------')


            # If examples_vi is empty : What I understand from this is that there is not tuple
            """
            When examples_vi is empty, it means there is not tuple. But,
            """


            if examples_vi.empty:

                # Then below this new branch add a leaf node with label = most common value of Target_attribute in Examples
                target_value, target_count = np.unique(target_attribute, return_counts=True)
                most_frequent_value = target_value[np.argmax(target_count)]


                print("**** CONDITION 5 - examples_vi is empty ****")
                print('returning root = ', root , ', most_frequent_value = ', most_frequent_value, ', for prev_node = ', prev_node)

                self.leaf_nodes.append( {prev_node: most_frequent_value })

                #labels.append(most_frequent_value)
                return root, most_frequent_value

            else:

                """
                change this to recursive run call
                """

                attributes_ = attributes

                print("**** CONDITION 5 - examples_vi is NOT empty ****")
                print('------------------------------------------')
                print('attributes_ before removing : ', attributes_)
                #attributes.remove(A)

                print('attribute A to be removed : ', A)

                if A in attributes_:
                    attributes_.remove(A)
                print('attributes_ after removing : ', attributes_)


                print('------------------------------------------')

                print('------------------------------------------')


                # get the corresponding target attribute tuples


                print('corresponding tuple index : ', examples_vi.index)

                #print('corresponding target_attribute below : ')

                indices = examples_vi.index


                target_attribute_ = target_attribute.loc[ indices ]


                #display(target_attribute_)


                """
                chi-2 on the id3

                """


                # get actual_table

                #actual_table = pd.crosstab(examples_vi[], target_attribute_)

                #= pd.crosstab(data[chosen_feature], data['target'])



                self.run(examples_vi, target_attribute_ ,attributes_,  str(A) + "-" + str(vi)  ) # remove not in list error
                #sys.exit(0)

                print()

                print('vi : ', vi )
                print('------')



            vi_with_root = []


            """
            I am thinking of using dictionary as for the mean branching an attribute

            - Root is a list

            """

        print('root, out of the loop : ', root)
        self.tree.append(root)



    def compute_impurity_by_label(self, attribute, impurity='entropy'): # Impurity of the total dataset : DONE

        """
        FEATURES:

        attribute : pandas df
            the column whose entropy is to be calculated

        impurity : string
            the impurity measure used- gini or entropty


        Returns
            np real scalar number
        """


        # get the total number of instances/rows in the dataset
        N = attribute.shape[0]

        #print('\t\t Number of rows in attribute param:', N)
        #sys.exit(0)

        # get the count
        label_values, label_counts = np.unique(attribute, return_counts=True)
        label_fractions = []


        # get the fractions for the each of the labels- better to use loop be cause there can be more than two labels

        for count in label_counts :
            #print(Decimal(count/N))

            result_float = float( count/ Decimal(N) )

            label_fractions.append( result_float  )


        #print('\t\tlabel_fractions: ',label_fractions)

        label_fractions = np.array( label_fractions )
        #print('\t\tDifferent label values collected: ', label_values)
        #print('\t\tDifferent label counts colleceted: ', label_counts)
        #print('\t\tFractions of different labels: ', label_fractions)


        # write a subroutine for entropy
        if impurity=='entropy':

            return -np.sum(  label_fractions * np.log2(label_fractions) )


        # write a subroutine for gini
        elif impurity=='gini':

          return 1 - np.sum(  np.square( label_fractions )   ) # 1 - sum of elementwise fraction #This returns the complete gini


        if impurity == 'max_value_error':
            return 1 - np.max(label_fractions)


        else :

            print("ERROR: impurity metric can be either of gini or entropy.")
            return -1


    def information_gain_for_discrete_attribute(self, examples_a, target_attribute, impurity='entropy'): # 02/28/2024 This stays. Fix this
        """

        Parameters
        ----------
        examples_a : the attribute column whose feature is to be calculated
            type: Pandas Series

        target_attribute : attribute whose value is to be predicted by tree
            type: Pandas Series

        attribute : attribute/column name for examples_a
            type: string

        impurity_measure : gini/entropy
            type: string

        Returns
        -------
        scalar real number



        self.information_gain( examples[a], target_attribute, 'entropy') # examples is pd df

        """

        print('--------------- info gain for discrete attr method -------------------')
        #print('\t\t\t examples_a :\n', examples_a)
        #print('\t\t\t examples_a.shape :\n', examples_a.shape)



        #print('\t\t\t target_attribute :\n', target_attribute)




        #impurity_for_target_attribute = self.compute_impurity_for_discrete_attribute(target_attribute, impurity=impurity)


        # get the unique values in examples_a
        examples_a_values = np.unique(examples_a)

        N = examples_a.shape[0]

        result = self.compute_impurity_by_label(  attribute=target_attribute , impurity=impurity)

        #print( '\t\t\tresult after initialization : ', result) # ok

        #sys.exit(0)
        for a in examples_a_values:

            # get the subset of examples_a and corresponding tuple in target_attribute
            #examples_a[attribute]
            #print( examples_a[examples_a==a])
            #print('-----')
            #print('feature subset shape:\n', examples_a[examples_a==a].shape)
            #print('-----')

            #print( 'target subset shape:\n', target_attribute[examples_a==a].shape )


            #examples_a_subset = np.array( examples_a[examples_a==a] )
            """
            I don't need the line above rn
            """


            #target_a_subset = np.array( target_attribute[examples_a==a] ) # converting to np for faster computation

            n = target_attribute[examples_a==a].shape[0]
            #compute_impurity_by_label(  np.array( target_attribute[examples_a==a] ), impurity=impurity)


            prob_float = float( n/ Decimal(N) )


            impurity_a = self.compute_impurity_by_label( attribute=target_attribute[examples_a==a] ,
                                                        impurity=impurity) * prob_float

            result = result - impurity_a

            #print('\t\t---------------\t\t\n')



            #print('\t\t\t--- final info gain : ', result )

        print('============= info gain for discrete attr method =============')
        return result # returns a scalar real number

class RandomForest:

    def __init__(self) -> None:
        self.Forest = []
        self.treest = []
        self.leaves = []

    def rfTrees(self, X, Y, ntree):

        """

        Parameters
        ----------
        X : training examples
            pandas df

        Y : attriburte whose value is to be predicted by tree
            pandas series

        attributes : list of other attributes to be tested
            list w/ strings

        ntree : number of decision trees in random forest

        Result -> This function appends each decision tree to an array called 'Forest'

        AMENDMENT: The input parameter should look as follows:
            X, Y, attributes, ntree
        """

        for _ in range(ntree):

            tree = ID3(do_chi_test=True)

            #print('My Attr:',self.attributes)

            #Bootstraping the data
            index = self.bootStrap(X)

            #forming a decision tree for each sample
            print('Data:', X.iloc[index], Y.iloc[index])
            #print('My Attr:',X.iloc[index].columns.tolist())
            tree.run(X.iloc[index], Y.iloc[index], X.iloc[index].columns.tolist())

            #l = pd.DataFrame(self.leaves)

            self.treest.append(tree.tree)
            self.leaves.append(tree.leaf_nodes)

            #appending the tree to forest array
            self.Forest.append(tree)

    def bootStrap(self, X):

        """
        Parameters
        ----------
        X : training examples
            pandas df

        Returns
            indexes for randomly selected data.
        """

        #Choosing random data
        ind = np.random.choice(X.shape[0], size= X.shape[0], replace= False)

        return ind

    def predict(self, X):
        """
        Parameters
        ----------
        X : training examples
            pandas df

        Returns
            An array of predicted values(maximum voted predictions) from random forest for a data
        """
        t = pd.DataFrame(self.treest)
        l = pd.DataFrame(self.leaves)
        t.index = ['tree1','tree2','tree3']
        l.index = ['tree1','tree2','tree3']
        t.to_csv('/content/drive/My Drive/treeStruct.csv', header=False, lineterminator='\n\n')
        l.to_csv('/content/drive/My Drive/leafNodes.csv', header=False, lineterminator='\n\n')

        #Getting prections of given data from each tree and swapping the axes to get group the
        #preictions from same row in to one array.
        tree_pred = np.swapaxes(np.array([tree.predict(X) for tree in self.Forest]), 0, 1)

        #print('self.Forest: ',self.Forest)
        #tree_pred = pd.DataFrame(np.array([tree.predict(X) for tree in self.Forest]))

        #print('tree_pred:',tree_pred)

        #Getting maximum vote for a predicted value
        #forest_predictions = np.array([np.bincount(pred).argmax() for pred in tree_pred])

        forest_predictions = np.array([np.unique(pred, return_counts=True)[0][np.argmax(np.unique(pred, return_counts=True)[1])] for pred in tree_pred])

        return forest_predictions
        #tree_pred.to_csv('/content/drive/My Drive/predictionOut.csv',index=False, header=False)


df = pd.read_csv('/content/drive/My Drive/TrainData_8k.csv')
#dftest = pd.read_csv('/content/drive/My Drive/Test1_1000.csv')
#dftest = dftest.iloc[:, 1:]
df.drop(columns='TransactionID' , inplace=True)
df = df.sample(frac = 1)
#dftest.drop(columns='TransactionID' , inplace=True)
#attributes = df.columns.tolist()
#attributes.remove('isFraud')

a_ = ['ProductCD', 'card1', 'card4',
       'card6', 'addr1', 'addr2', 'TransactionDT', 'TransactionAmt',
       'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
       'C12', 'C13', 'C14',]

#X = df.iloc[:, :-1]
#y = df.iloc[:, -1]

X = df[a_]
y = df['isFraud']

#X = df.drop(columns=['isFraud']).values
#y = df['isFraud'].values
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, stratify=y, random_state=45)

#print('traindf', train_df)
"""
tree = RandomForest()
tree.rfTrees(X, y, 1)
output = pd.DataFrame()
output['TransactionID'] = dftest.get('TransactionID')
output['isFraud'] = tree.predict(dftest.iloc[:, 1:])
output.to_csv('/content/drive/My Drive/output.csv', index = False)
"""


# training
start_time = time.time()

tree_1 = RandomForest()
tree_1.rfTrees(X_train, y_train, 3)

end_time = time.time()

print('train time : ', start_time - end_time)



# testing
start_time_predict = time.time()

y_pred = tree_1.predict(X_val)

end_time_predict = time.time()

print('train time : ', end_time - start_time)
print('test time : ', end_time_predict - start_time_predict)

output = pd.DataFrame()
output['y_val'] = y_val
output['y_test'] = y_pred
output.to_csv('/content/drive/My Drive/max_value_error_test.csv', index = False)


#pd.DataFrame(pd.concat([y_pred, y_val], axis=1) , columns=['y_test', 'y_val']).to_csv('/content/drive/My Drive/entropy_test.csv')