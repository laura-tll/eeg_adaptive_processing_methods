#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:13:39 2021

@author: l.tilly
"""


import warnings
warnings.filterwarnings('ignore')  # "always" "error", "ignore", "always", "default", "module" or "once"

import pickle
# import matplotlib.pyplot as plt
import numpy as np 
# import decimal
# import scipy.io
# import mne
# from mne.preprocessing import ICA
# from mne.time_frequency import psd_multitaper

# from scipy.interpolate import LinearNDInterpolator

# import PIL
# from PIL import Image

import pandas as pd
import os

####################### Classifiers ############################
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, log_loss
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC, NuSVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.metrics.scorer import make_scorer



classifiers = [
        # KNeighborsClassifier(),    
        # NuSVC(probability=True), #NOT USING
        # DecisionTreeClassifier(),
        # RandomForestClassifier(), #NOT USING
        # AdaBoostClassifier(),
        # GradientBoostingClassifier(),  #NOT USING
        # GaussianNB(),
        LinearDiscriminantAnalysis()
        # QuadraticDiscriminantAnalysis(), 
        # SVC(probability=True)
    ]

tuned_parameters = [
        # {'n_neighbors':[2, 3, 4, 5, 6, 7, 8, 9, 10], 'weights' : ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
        # # {'kernel': ['linear', 'poly', 'rbf'], 'gamma': [0.1,0.15,0.2,0.5,1,1.5]},
        # {'max_depth': [ 5, 10, 15, 20, 25] },
        # # {'max_depth': [ 5, 10, 15], 'n_estimators': [1, 3, 5, 7, 9], 'max_features': [1, 2, 3, 4, 5] },
        # {'n_estimators': [20, 30, 40, 50, 60, 70, 80] },
        # # {"loss":["deviance"], "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],"min_samples_split": np.linspace(0.1, 0.5, 12), "min_samples_leaf": np.linspace(0.1, 0.5, 12),"max_depth":[3,5,8],"max_features":["log2","sqrt"],"criterion": ["friedman_mse",  "mae"],"subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0]},
        # {None},
        {'solver': ["lsqr","eigen"]},
        # {'reg_param' : [0.0, 1.0, 2.0, 3.0]},
        # {'kernel': ['linear', 'poly', 'rbf'], 'gamma': [1e-1, 1.5e-1, 2e-1],'C': [0.5, 1, 2] }
    ]


score = 'accuracy'
fInit = "log_erp_inter_traditional_"

erp = "/home/dcas/l.tilly/Documents/data/ERP_data_2sec_processed.pickle"
pickle_in = open(erp, "rb")
file = pickle.load(pickle_in)
pickle_in.close()

# file_erp = file[:,:,:,:,:,:].copy()

psd_top = [0,  1,  2,  3,  5,  6,  9, 10, 13, 14, 15, 16, 17, 19, 23, 24, 26, 27, 28, 29]

# lw_features = np.zeros([30,20])
# hw_features = np.zeros([30,20])

for classifier, parameters in zip(classifiers, tuned_parameters):   
    # Logging for Visual Comparison
    log_cols=["Classifier", "Accuracy", "Log Loss"]
    log = pd.DataFrame(columns=log_cols)

    clf = classifier
    name = clf.__class__.__name__
    
    print()
    print()
    print('Processing with Classifier:', name)
    if os.path.exists("./output/" +fInit +name +".pickle") == True:
        print("File exists:", "./output/" +fInit +name +".pickle")
        continue
    
    for i in range(13):
        print("Testing with participant:", i)

        data         = np.zeros([1*2*2*15, 20])
        labels         = []
        counter_test   = 0
 
        for p in range(13):
            for l in range (2):
                for s in range (2):
                    for e in range (15):
                         
                        if p == i:
                            erp = []
                            for c in psd_top :
                                erp.append(file[p,l,s,e,c,2048:2663].mean(0))
                            data[counter_test, :] = erp
                            labels.append(l)
                            counter_test += 1

        data = np.array(data)
        labels = np.array(labels)
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)   
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
            

 
                   
        if parameters != {None}:            
                   print("# Tuning hyper-parameters for %s" % name)
                   clf = GridSearchCV(classifier, parameters, cv=5, scoring=score, n_jobs=-1, verbose=0)
           
        best_param = None
        clf.fit(X_train, y_train)  
        if parameters != {None}:
            best_param = clf.best_params_
            print("Best parameters for:", name)
            print(clf.best_params_)        
            
        # if name == "KNeighborsClassifier":
        #     clf = KNeighborsClassifier(n_neighbors=best_param['n_neighbors'], weights=best_param['weights'], algorithm=best_param['algorithm'])
        # elif name == "NuSVC":
        #     clf = NuSVC(kernel='poly', gamma=1, probability=True)
        # elif name == "DecisionTreeClassifier":
        #     clf = DecisionTreeClassifier(max_depth=best_param['max_depth'])
        # elif name == "RandomForestClassifier":
        #     clf = RandomForestClassifier(max_depth=best_param['max_depth'], n_estimators=best_param['n_estimators'], max_features=best_param['max_features'])
        # elif name == "AdaBoostClassifier":
        #     clf = AdaBoostClassifier(n_estimators=best_param['n_estimators'])
        # elif name == "GradientBoostingClassifier":
        #     clf = GradientBoostingClassifier(loss=best_param['loss'], learning_rate=best_param['learning_rate'], min_samples_split=best_param['min_samples_split'], min_samples_leaf=best_param['min_samples_leaf'], max_depth=best_param['max_depth'], max_features=best_param['max_features'], criterion=best_param['criterion'], subsample=best_param['subsample'])
        # elif name == "GaussianNB":
        #     clf = GaussianNB()
        if name == "LinearDiscriminantAnalysis":
            clf = LinearDiscriminantAnalysis(solver=best_param['solver'])
        # elif name == "QuadraticDiscriminantAnalysis":
        #     clf = QuadraticDiscriminantAnalysis(reg_param=best_param['reg_param'])
        # elif name == "SVC":
        #     clf = SVC(kernel=best_param['kernel'], C=best_param['C'], gamma=best_param['gamma'], probability=True)                
        
        clf.fit(X_train, y_train) 
        
        scoring = {'prec_macro': 'precision_macro',
           'rec_macro': make_scorer(recall_score, average='macro'),
           'f1_macro': 'f1_macro'}
        ## Cross validation for the best parameters
        scores = cross_validate(clf, data, labels, scoring=scoring, cv=10, return_train_score=False)
        #print(sorted(scores.keys()))
        
        print("Precision: %0.2f (+/- %0.2f)" % (np.mean(scores['test_prec_macro']), np.std(scores['test_prec_macro'])*2) )
        print("Recall: %0.2f (+/- %0.2f)" % (np.mean(scores['test_rec_macro']), np.std(scores['test_rec_macro'])*2) )
        print("F1-score: %0.2f (+/- %0.2f)" % (np.mean(scores['test_f1_macro']), np.std(scores['test_f1_macro'])*2) )     
        
        
        
        # for tpm in range(len(scores['test_prec_macro'])):
        log_entry = pd.DataFrame([[i, scores['test_prec_macro'].mean()*100, 0]], columns=log_cols)
        log = log.append(log_entry)  
            
        print("="*30)
        print("="*30)
        print()
                    
    pickle_out = open("/home/dcas/l.tilly/Documents/data/lh_erp_intra_tradi.pickle", "wb")
    pickle.dump(log, pickle_out)
    pickle_out.close()
                
        
val = log.loc[:,['Accuracy']].values
print(val.min(), val.max(), val.mean(), np.median(val))



   
                                    
                