#%%
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 15:32:08 2020

@author: mperaud


#!/usr/bin/python

"""





# =============================================================================
#     EXPLORING ENRON DATASET
# =============================================================================






import numpy as np
import pickle
import sys
sys.path.append("../tools/")

from tester import dump_classifier_and_data

file_path = "../final_project/final_project_dataset.pkl"
data_dict = pickle.load(open(file_path, "rb"), fix_imports=True)

#data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

import matplotlib
from matplotlib import pyplot as plt
#plt.rcParams['backend'] = "Qt4Agg"
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

import pandas as pd
import seaborn as sb


from io import StringIO
from tabulate import tabulate

#pd.set_option('display.width', 500) ## in order to get nicer display of pandas dataframes on screen
#pd.options.display.float_format = '{:,.0f}'.format




print("            ")



# =============================================================================
# SIZE, AVAILABLE FEATURES
# =============================================================================

Enron_df = pd.DataFrame.from_records(data_dict).T
print("Features_list :")
Enron_df.info()


# =============================================================================
#   POIs in the DATASET
# =============================================================================


print("    ")
print("NUMBER OF POI IN THE SAMPLE: ",Enron_df['poi'].value_counts())

POI=()
POI=(Enron_df[Enron_df['poi']== True])
print(tabulate(POI.iloc[0:-1,16:17], headers='keys', tablefmt='presto',floatfmt=".0f"))




# =============================================================================
# DATASET CLEANING - Numeric values as numbers
# =============================================================================

clefs=Enron_df.columns.tolist()
for i in clefs :
    if i !='email_address':
        Enron_df[i]=Enron_df[i].astype(float)


# =============================================================================
# DATASET CLEANING - NAN values in features
# =============================================================================

valeursnan = Enron_df.isnull().sum()


ax=valeursnan.plot.barh()
for i, v in enumerate(valeursnan):
    ax.text(v + 3, i + .25, str(v), color='darkblue', fontweight='bold')
plt.title('Number of incomplete data per feature, out of 146 records', fontsize=16)
plt.xlabel('number of null values', fontsize=12)
plt.ylabel('features', fontsize=12)
plt.show()


# =============================================================================
# DATASET CLEANING ENQUIRY ABOUT HIGH NUMBER OF NAN VALUES FEATURES
# =============================================================================



print('non null restricted_stock_deferred')

print(tabulate(Enron_df.query('restricted_stock_deferred >0 or restricted_stock_deferred<0').iloc[0:-1,7:8], headers='keys', tablefmt='fancy_grid'))
print('non null loan advances')
print(tabulate(Enron_df.query('loan_advances>0 or loan_advances<0').iloc[0:-1,4:5], headers='keys', tablefmt='fancy_grid',floatfmt=".0f"))
#print(tabulate(Enron_df[Enron_df['loan_advances']<0].iloc[0:-1,5:6], headers='keys', tablefmt='fancy_grid',floatfmt=".0f"))
print('non null director fees')
print(tabulate(Enron_df[Enron_df['director_fees']>0].iloc[0:-1,20:21], headers='keys', tablefmt='fancy_grid',floatfmt=".0f"))



# =============================================================================
# DATASET CLEANING - REPLACE ALL NAN VALUES IN NUMERIC FEATURES BY 0 
# =============================================================================


Enron2=Enron_df.fillna(0)
Enron3=Enron2.T
print(Enron2.info()) ## check that all data are now numeric, except email addresses


# =============================================================================
# DATASET CLEANING - IDENTIFY WRONG ENTRIES AND CORRECT
# =============================================================================

print("Check totals and extract records with wrong entries")
Enron2['financialdata']=Enron2['bonus']+Enron2['deferral_payments']+Enron2['deferred_income']+Enron2['director_fees']
Enron2['financialdata']=Enron2['financialdata']+Enron2['expenses']+Enron2['salary']
Enron2['financialdata']=Enron2['financialdata']+Enron2['loan_advances']+Enron2['long_term_incentive']+Enron2['other']
Enron2['error']= Enron2['financialdata']- Enron2['total_payments']
print(tabulate(Enron2.query('error >0 or error <0').T, headers='keys', tablefmt='fancy_grid',floatfmt=".0f")) 



# =============================================================================
# #CORRECT BHATNAGAR SANJAY DATA
# =============================================================================


Enron2.iloc[11,1]=0#to_messages
Enron2.iloc[11,2]=0#bonus
Enron2.iloc[11,3]=137864#total_payments
Enron2.iloc[11,7]=-2604490#restricted_stock_deferred
Enron2.iloc[11,9]=15456290#total_stock_value
Enron2.iloc[11,10]=137864#expenses
Enron2.iloc[11,12]=15456290#exercised_stock_options
Enron2.iloc[11,14]=0#other
Enron2.iloc[11,19]=2604490#restricted_stock
Enron2.iloc[11,20]=0#director_fees



# =============================================================================
# #CORRECT BELFER ROBERT │ DATA
# =============================================================================

Enron2.iloc[8,2]=0#deferral_payments 
Enron2.iloc[8,8]=-102500#deferred_income
Enron2.iloc[8,20]=102500#director_fees
Enron2.iloc[8,12]=0#exercised_stock_options 
Enron2.iloc[8,10]=3285#expenses 
#Enron2.iloc[8,14]=0#other 
Enron2.iloc[8,19]=44093#restricted_stock
Enron2.iloc[8,7]=-44093#restricted_stock_deferred  
Enron2.iloc[8,3]=3285#total_payments
Enron2.iloc[8,9]=0#total_stock_value







# =============================================================================
#     REMOVE COLUMN 'LOAN_ADVANCES', 'ERROR','FINANCIAL DATA'
# =============================================================================


Enron2.drop(axis=1, labels=['loan_advances','error','financialdata'], inplace=True)

# =============================================================================
# DATASET CLEANING - IDENTIFY INCOMPLETE LINES
# =============================================================================

print("Find records where NAN datas are more than 18 features out of 21")
notrecorded = []
for person in Enron3:
    
    n = 0
    for key, value in Enron3[person].iteritems():
        if value == 'NaN' or value == 0:
            n += 1
       
        if n > 18:
            if person not in notrecorded :
                notrecorded.append(person)


for nom in notrecorded :
        print("incomplete lines: ", Enron2.loc[nom][['poi','total_payments']])               



# =============================================================================
# REMOVE INCOMPLETE LINES FROM DATASET
# =============================================================================

print(Enron2.tail(26))
Enron2.drop(labels='GRAMM WENDY L',axis=0, inplace=True)
Enron2.drop(labels='LOCKHART EUGENE E',axis=0,inplace=True)
Enron2.drop(labels='WHALEY DAVID A',axis=0,inplace=True)
Enron2.drop(labels='WROBEL BRUCE',axis=0,inplace=True)

Enron2.drop(labels='THE TRAVEL AGENCY IN THE PARK',axis=0, inplace=True)
Enron3=Enron2.T



#
# =============================================================================
# 
# DATASET CLEANING - FIND OUTLIERS (1/2)
# First identify outliers (small an enormous values)
# =============================================================================

sb.lmplot(x='bonus', y= 'salary', data=Enron2, palette='Set1',height=5)
plt.title('Salary/Bonus', fontsize=18)
plt.xlabel('Bonus', fontsize=16)
plt.ylabel('Salary', fontsize=16)
plt.show()


print("bigest salaries, bonus, with POI indication")
outhigh=Enron2[Enron2['bonus']>1000000 ].iloc[0:-1,[0,4,15]]

print(tabulate(outhigh.sort_values(by=['bonus','salary'],ascending=False), headers='keys', tablefmt='fancy_grid',floatfmt=".0f"))
Enron2.drop(labels='TOTAL',axis=0, inplace=True)#remove TOTAL from dataset
#print("zero salaries")
#print(tabulate(Enron2[(Enron2['salary']==0)][['salary','bonus','poi','other','total_payments']], headers='keys', tablefmt='fancy_grid',floatfmt=".0f"))




# =============================================================================
# DATASET CLEANING - Find outliers (2/2)
# See new graph after remove of non significant datapoints
# =============================================================================

sb.lmplot(x='salary', y= 'bonus', hue='poi', data=Enron2, palette='bright',height=8,markers=['P','o'])
plt.title('salary/bonus for POI and non-POI', fontsize=18)
plt.xlabel('salary', fontsize=16)
plt.ylabel('bonus', fontsize=16)
plt.show()

Enron2.drop(labels='LAVORATO JOHN J',axis=0,inplace=True)
Enron2.drop(labels='KITCHEN LOUISE',axis=0,inplace=True)

# =============================================================================
# ADDITIONAL FEATURES CREATION
# =============================================================================

# function for any ratio
def ratiocomp(x, y) :
    if x == 0 or y == 0:
        ratio = 0
    else :
        ratio= float(x) / float(y)
        
    return ratio

# =============================================================================
# # add new features in Enron2 dataset : ratio bonus/salary and from /to POIs
# =============================================================================

for i in Enron3:
    
    row = Enron3[i]
    
    x= row['bonus']
    y = row['salary']
    bonus_salary_ratio= ratiocomp(x, y)
    Enron3.loc['ratio_bonus_salary',i] = bonus_salary_ratio
    

    rec = row['from_messages']
    poif = row['from_poi_to_this_person']
    receivedfromPOI_ratio= ratiocomp(poif, rec)
    Enron3.loc['part_from_POI',i] = receivedfromPOI_ratio
    
    sent = row['to_messages']
    pois = row['from_this_person_to_poi']
    senttoPOI_ratio= ratiocomp(pois, sent)
    Enron3.loc['part_to_POI',i] = senttoPOI_ratio
    
Enron2=Enron3.T

# =============================================================================
# CREATE ENRON4, where all strings disappear
# =============================================================================
"""
Replace current index by numeric index and drop non numeric feature (email_address)
"""
Enron4=Enron2.reset_index(drop=True)
Enron4.drop('email_address',axis=1, inplace=True)

"""
matrix transpose has moved all figures into strings, those two lines to get back to numeric values
"""
for col in Enron4.select_dtypes('object'):
    Enron4[col]=Enron4[col].astype('float')

#print(tabulate(Enron4.iloc[0:-1,0:5], headers='keys', tablefmt='fancy_grid'))# check that shape of the dataset is correct
#print(tabulate(Enron4.iloc[0:-1,9:17], headers='keys',tablefmt='fancy_grid'))
#print(tabulate(Enron4.iloc[0:-1,0:6], headers='keys',tablefmt='fancy_grid'))
# =============================================================================
# CREATING TRAIN AND TEST SETS FROM CLEANED DATABASIS, WITH train_test_split TOOL FROM SKLEARN
# =============================================================================



features_list1 =[ 'part_from_POI','part_to_POI','ratio_bonus_salary','bonus', 'deferral_payments', 'deferred_income', 'director_fees', 'exercised_stock_options', 'expenses', 
                'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 'long_term_incentive', 'other', 'salary', 'total_payments', 'total_stock_value']



    
from sklearn.model_selection import train_test_split
trainset, testset = train_test_split(Enron4, test_size=0.3,random_state=42)

"""
CHECK trainset and testset contents and size, as well as % of POI in each SET
"""


a=trainset['poi'].sum()
b=testset['poi'].sum()

c=trainset['poi'].sum()/len(trainset)
d=testset['poi'].sum()/len(testset)
print("number of POIs in trainset :",a, "out of",len(trainset),"that is in % :",c)
print("number of POIs in testset :",b, "out of",len(testset),"that is in % :",d)

# =============================================================================
# SET FEATURES LISTS (that will be modified during various tests)
# =============================================================================



#feature_list4 is the list of all features except those with negative values
features_list4 =[ 'part_from_POI','part_to_POI','ratio_bonus_salary','bonus', 'deferral_payments', 'director_fees', 'exercised_stock_options', 'expenses', 
                'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 'long_term_incentive', 'other', 'salary', 'total_payments', 'total_stock_value']


features_list0 =['part_from_POI','ratio_bonus_salary','from_messages','from_poi_to_this_person']

# ============================================================================================
# Create function prepa that prepares train set and test sets with parametric list of features
# ============================================================================================

def prepa (frame,features_list):
    X=frame[features_list]
    y=frame['poi']
   
    return X, y

X_train,y_train=prepa(trainset,features_list4)
X_test,y_test=prepa(testset,features_list4)



# =============================================================================
# CREATE EVALUATION TOOL called eval
# function "eval(clf), will train any skitlearn model, with same trainset and dataset,
# using a given features list, and produce
# confusion matrix
# classification report, with precision, recall, f1-score, and accuracy and it will print a learning curve
# It will also print "weight of each feature" in model DecisionTreeClassifier ()
#(sklearn currently provides model-based feature importances for tree-based models and linear models.
# However, models such as e.g. SVM and kNN don't provide feature importances.)
# =============================================================================

import warnings##to avoid messages about division by 0 when recall is 0 
import sklearn.exceptions##idem

from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, confusion_matrix,classification_report
from sklearn.model_selection import learning_curve
import scikitplot as skplt
from sklearn import tree
import graphviz
from graphviz import Source
from sklearn.feature_selection import SelectKBest, chi2, f_classif



def eval(xxx) :
    clf = xxx
    clf=clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    
    print("\n","CONFUSION MATRIX","\n",confusion_matrix(y_test,y_pred),"\n")
    print("\n\n")
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=False)
    Titre="model :"+str(clf)+"\n"+str(X_train.columns)
    plt.title(Titre)
    print("CLASSIFICATION REPORT","\n",classification_report(y_test, y_pred))
    
    return clf
  
        

print("FIRST EVALUATION OF DATASET")
X_train,y_train=prepa(trainset,features_list4)
X_test,y_test=prepa(testset,features_list4)
eval(DecisionTreeClassifier())
clf = DecisionTreeClassifier()
resultat=clf.fit(X_train,y_train)
pd.DataFrame(clf.feature_importances_,index=(X_train.columns)).plot.bar()
plt.title("Importance of features in list :"+str(X_train.columns)+"with "+str(clf))




"""
https://scikit-learn.org/stable/modules/tree.html#tree => creates a pdf file !
"""
dt_target_names = [str(s) for s in y_train.unique()]  
dot_data = tree.export_graphviz(resultat, out_file=None,feature_names=X_train.columns,class_names=dt_target_names,rounded=True,leaves_parallel=(True))
                            
graph = graphviz.Source(dot_data) 
color = '#{:02x}{:02x}'
graph.render("Enron_decision_tree") 

#%%
# =============================================================================
# USE EVALUATION TOOL on
# SVM, Kneigbor, Classifier
# and 4 different features lists
# =============================================================================

def testboucle (liste,model):
    X_train=[]
    y_train=[]
    X_test=[]
    y_test=[]
    
    X_train,y_train=prepa(trainset,liste)
    X_test,y_test=prepa(testset,liste)
    clf=model
    print("\n\n",model,"MODEL EVALUATION WITH FEATURE LIST :\n",liste)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    
    print("\n","CONFUSION MATRIX","\n",confusion_matrix(y_test,y_pred),"\n")
    print("\n\n")
    print("CLASSIFICATION REPORT","\n",classification_report(y_test, y_pred))
    
    Titre="Model :"+ str(clf)+"\n"+str(X_train.columns)
    plt.title(Titre, loc='center', pad=None)
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=False)
    plt.title(Titre)   
   
    
    N, train_score, val_score = learning_curve(clf, X_train, y_train,
                                                 cv=4,scoring='f1', train_sizes=np.linspace(0.1,1,10))
    plt.figure(figsize=(12,8))
    plt.plot(N,train_score.mean(axis=1), label='train_score')
    plt.plot(N,val_score.mean(axis=1), label='validation_score')
    plt.legend()
    plt.title(Titre)
    
    return print("\n")

# =============================================================================
# FIRST VIEW Select features_list for DecisionTreeClassifier with  feature_importance and with SelectKbest
# This allows proving utility of new features Created
# =============================================================================


print("SELECT BEST FEATURES LIST DecisionTreeClassifier")

liste =features_list4
X_train=[]
y_train=[]
X_test=[]
y_test=[]
    
X_train,y_train=prepa(trainset,liste)
X_test,y_test=prepa(testset,liste)
clf=DecisionTreeClassifier()
clf.fit(X_train,y_train)




plt.title("Importance of features in list :"+str(X_train.columns))
pd.DataFrame(clf.feature_importances_,index=(X_train.columns)).plot.bar()


clf.fit(X_train,y_train)
#y_pred=clf.predict(X_test)
    
Selector = SelectKBest(f_classif,k=4)
Selector.fit_transform(X_train,y_train)
selection=Selector.get_support()
for a,b in zip(features_list4,selection) :
    if b == True :
        print (a)


# =============================================================================
# DecisionTreeClassifier : chose best parameters and best features selection
# with SelectKbest and GridSearchCV :
# First create pipeline for cross validation
# Then GridSearchCV for best parameters
# With SelectKbest for optimized number of features        
# =============================================================================
print("\n\n SELECT best HYPERPARAMETERS")

liste =features_list4
X_train=[]
y_train=[]
X_test=[]
y_test=[]
    
X_train,y_train=prepa(trainset,liste)
X_test,y_test=prepa(testset,liste)


from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.pipeline import make_pipeline
warnings.filterwarnings('ignore', 'Solver terminated early.*')

pipe = Pipeline([('scaler', MinMaxScaler()),
                 ('selector', SelectKBest(f_classif, k=4)),
                 ('classifier', DecisionTreeClassifier())])

search_space = [{'scaler':[StandardScaler(),MinMaxScaler(),RobustScaler()]},
                {'selector__k': [4,5,6,7]},       
                
  
                {'classifier':[KNeighborsClassifier()],
                 'classifier__n_neighbors':[1,3,5,7],
                 'classifier__weights':['uniform','distance'],              
                 'classifier__metric':['euclidean','manhattan'],
                 'classifier__leaf_size':[2,8,10,20,30]},
                
                {'classifier':[svm.SVC()],
                'classifier__C': [1, 50, 100, 1000],
                'classifier__gamma': [0.5, 0.1, 0.01],
                'classifier__degree': [1, 2],
                'classifier__kernel': ['rbf', 'poly', 'linear'],
                'classifier__max_iter': [1, 100, 1000]},
                
                
                {'classifier':[DecisionTreeClassifier()],
                 'classifier__criterion' :['gini','entropy'],
                 'classifier__splitter' : ['best','random'],
                 'classifier__min_samples_split':[2,3,4,5,6],
                 'classifier__max_depth':[1,2,3]},]


clf = GridSearchCV(pipe, search_space, cv=3, verbose=0,scoring ='recall')
clf = clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

print(clf.best_estimator_)
print(clf.best_score_)
print("\n","CONFUSION MATRIX","\n",confusion_matrix(y_test,y_pred),"\n")



# =============================================================================
# IDENTIFY 4 BEST PARAMETERS WITH SVC
# =============================================================================
print("BEST_FEATURES with BEST MODEL")

liste =features_list4
X_train=[]
y_train=[]
X_test=[]
y_test=[]
  
X_train,y_train=prepa(trainset,liste)
X_test,y_test=prepa(testset,liste)
clf=make_pipeline(MinMaxScaler(),svm.SVC(C=1000, degree=1, gamma=0.5, max_iter=1))
clf=clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)


Selector = SelectKBest(f_classif,k=4)
Selector.fit_transform(X_train,y_train)
selection=Selector.get_support()
for a,b in zip(features_list4,selection) :
    if b == True :
        print(a)

# =============================================================================
# BEST SELECTION (model + features + hyperparameters)
# =============================================================================

print("FINAL MODEL")
liste =features_list0
X_train=[]
y_train=[]
X_test=[]
y_test=[]
  
X_train,y_train=prepa(trainset,liste)
X_test,y_test=prepa(testset,liste)

clf=make_pipeline(MinMaxScaler(),svm.SVC(C=1000, degree=1, gamma=0.5, max_iter=1))
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

print("\n","CONFUSION MATRIX","\n",confusion_matrix(y_test,y_pred),"\n")

    
Titre="Model :"+ str(clf)+"\n"+str(X_train.columns)
plt.title(Titre, loc='center', pad=None)
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=False)
print("CLASSIFICATION REPORT","\n",classification_report(y_test, y_pred))   

N, train_score, val_score = learning_curve(clf, X_train, y_train,
                                                 cv=4,scoring='f1', train_sizes=np.linspace(0.1,1,10))
plt.figure(figsize=(12,8))
plt.plot(N,train_score.mean(axis=1), label='train_score')
plt.plot(N,val_score.mean(axis=1), label='validation_score')
plt.legend()
plt.title(Titre)