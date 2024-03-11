#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 12:11:08 2024

@author: shahriar
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split


import os
import time 


import sys

# Arithmetic 
from fractions import Fraction
from decimal import Decimal

class ID3: 
    
    def __init__(self):
        self.counter = 0 
        self.tree = []
        #self.labels = []
        self.leaf_nodes = []
    def print_tree_list(self): 
        print(self.tree)
    
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
            print('---> returning, root = ', root, ', target_values = ', target_values, ', for prev_node = ', prev_node)
            self.leaf_nodes.append( {prev_node: target_values}) 
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
            print('returning, root = ', root , ', most_frequent_value =', most_frequent_value , 'for prev_node = ', prev_node)
            
            
            self.leaf_nodes.append( {prev_node: most_frequent_value })
            #labels.append(most_frequent_value)
            
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
                                                                     impurity='entropy')        
            
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
            """
            
            print('root after adding attr values : ', root)
            
            print('----\n\n')
            
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
                
                
                
                self.run(examples_vi, target_attribute_ ,attributes_,  str(A + "-" + vi ) ) # remove not in list error 
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
        #self.labels.append(labels)
        #A = None 
        #examples_vi = [] #None 
        
        #branches = [] # the branches of the tree
        
        
        #if len( branches_vi) == 0: 
            # Then below this new branch add a  leaf node with label = most common value of Target_attribute in Examples
        
        
        #else:  # below this new branch add the subtree 
        #    ID3( examples_vi,  target_attribute, )            
    
      
        
    def compute_impurity_by_label(self, attribute, impurity='gini'): # Impurity of the total dataset : DONE
        
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
       
    def predict(self, X, tree):
        """
        Parameters
        ----------
        X : training examples
            pandas df 
        
        tree : reference of decision tree formed by ID3 

        Returns 
            predicted values of decision tree formed by ID3
        """
        return np.array([self.predictTree(x[1], tree) for x in X.iterrows()])
    
    def predictTree(self, X, tree, position=None, root = None):
        
        """
        Parameters
        ----------
        X : training example - single row
            pandas df 
        
        tree : reference of decision tree formed by ID3 

        Returns 
            predicted value of decision tree formed by ID3 - single row
        """
        if tree.tree:
            if root == None:
                subroot = list(tree.tree[::-1][0])
                branchValue = subroot[0] + '->' + X[subroot[0]]
                print('branchValue:',branchValue)
            else:
                print('tree.tree:',tree.tree[::-1][position])
                subroot = tree.tree[::-1][position]
                branchValue = root + '<-' + tree.tree[::-1][position][0] + '->' + X[tree.tree[::-1][position][0]]
        
            if branchValue in subroot:
                a = subroot[0] + '-' + X[subroot[0]]
                print('a:',a)
                for d in tree.leaf_nodes:
                    if a in d.keys():
                        print('a:',d.get(a)[0])
                        #return d.get(a)[0]
                        if d.get(a)[0] == 'Y':
                            print('a')
                            return 1
                        else:
                            return 0
            
                print('len(subroot):',len(subroot))
                print('subroot[::-1].index(branchValue):', subroot.index(branchValue))
                position = len(subroot) - subroot.index(branchValue)
                print('position:',position)
                print('tree:',tree.tree[::-1][position])
                self.predictTree(X, tree, position=position, root= subroot[0] + '-' + X[subroot[0]])
                
        
        return 1
                
        
        
             
class RandomForest:
    
    def __init__(self) -> None:
        self.Forest = []
    
    def rfTrees(self, X, Y, attributes, ntree):
        
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
            
            tree = ID3()
            
            #Bootstraping the data
            index = self.bootStrap(X)
            
            #forming a decision tree for each sample
            tree.run(X.iloc[index], Y.iloc[index], attributes)
            
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
        
        #Getting prections of given data from each tree and swapping the axes to get group the 
        #preictions from same row in to one array.
        tree_pred = np.swapaxes(np.array([tree.predict(X, tree) for tree in self.Forest]), 0, 1)
        print('tree_pred:',tree_pred)
        
        #Getting maximum vote for a predicted value
        forest_predictions = np.array([np.bincount(pred).argmax() for pred in tree_pred])
        
        return forest_predictions 
        
    



df = pd.read_csv('/Users/rahulpayeli/Documents/ML/test.csv')
df.drop(columns='ID' , inplace=True)
attributes = df.columns.tolist()
attributes.remove('PlayTennis')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

#X = df.drop(columns=['isFraud']).values
#y = df['isFraud'].values
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print('traindf', train_df)

tree = RandomForest()
tree.rfTrees(X, y, attributes,3)
predictions = tree.predict(X)
print('predictions:',predictions)
