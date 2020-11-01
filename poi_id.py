#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','bonus','exercised_stock_options']
 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
## identification of outliers done in Newdataset_evaluation - copie.py
data_dict.pop('GRAMM WENDY L',None)
data_dict.pop('LOCKHART EUGENE E',None)
data_dict.pop('WHALEY DAVID A',None)
data_dict.pop('WROBEL BRUCE',None)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',None)
   

##Correct Belfer 
data_dict['BELFER ROBERT']['deferral_payments']=0
data_dict['BELFER ROBERT']['deferred_income']=-102500
data_dict['BELFER ROBERT']['director_fees']=102500
data_dict['BELFER ROBERT']['exercised_stock_options']=0
data_dict['BELFER ROBERT']['expenses']=3285
data_dict['BELFER ROBERT']['other']=0
data_dict['BELFER ROBERT']['restricted_stock']=44093
data_dict['BELFER ROBERT']['restricted_stock_deferred']=-44093
data_dict['BELFER ROBERT']['total_payments']=3285
data_dict['BELFER ROBERT']['total_stock_value']=0

##Correct Bhatnagar
data_dict['BHATNAGAR SANJAY']['director_fees']=0
data_dict['BHATNAGAR SANJAY']['exercised_stock_options']=15456290
data_dict['BHATNAGAR SANJAY']['other']=0
data_dict['BHATNAGAR SANJAY']['expenses']=137864
data_dict['BHATNAGAR SANJAY']['restricted_stock']=2604490
data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred']=-2604490  
data_dict['BHATNAGAR SANJAY']['total_payments']=137864
data_dict['BHATNAGAR SANJAY']['other']=15456290

##Remove inconsistant data 'TOTAL'
data_dict.pop('TOTAL',None)  

##Remove "polluting" records
data_dict.pop('KITCHEN LOUISE',None)
data_dict.pop('LAVORATO JOHN J',None) 


### Task 3: Create new feature(s)
### create new features
### new features are: fraction_to_poi_email,fraction_from_poi_email

def dict_to_list(key,normalizer):
    new_list=[]

    for i in data_dict:
        if data_dict[i][key]=="NaN" or data_dict[i][normalizer]=="NaN":
            new_list.append(0.)
        elif data_dict[i][key]>=0:
            new_list.append(float(data_dict[i][key])/float(data_dict[i][normalizer]))
    return new_list


### create two lists of new features
part_to_poi=dict_to_list("from_poi_to_this_person","to_messages")
ratio_bonus_salary=dict_to_list("bonus","salary")

### insert new features into data_dict
count=0
for i in data_dict:
    data_dict[i]["part_to_poi"]=part_to_poi[count]
    data_dict[i]["ratio_bonus_salary"]= ratio_bonus_salary[count]

### Store to my_dataset for easy export below.
my_dataset = data_dict
features_list=['poi','bonus','exercised_stock_options','part_to_poi','ratio_bonus_salary']
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers (see in python 3 version and word doc)
from sklearn.neighbors import KNeighborsClassifier
clf =KNeighborsClassifier(leaf_size=2,metric='euclidean',n_neighbors=1,weights='uniform')

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html



# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)