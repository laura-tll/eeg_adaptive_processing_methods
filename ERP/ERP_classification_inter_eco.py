#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 13:15:48 2020

@author: Gaganpreet Singh
"""

import warnings
warnings.filterwarnings('ignore')  # "always" "error", "ignore", "always", "default", "module" or "once"

import numpy as np
import pandas as pd
import pickle
import os

####################### Classifiers ############################
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import preprocessing

        
#def warn(*args, **kwargs): pass
#import warnings
#warnings.warn = warn

#### Load EEG data
erp = "/home/dcas/l.tilly/Documents/data/ERP_data_2sec_processed.pickle"
pickle_in = open(erp, "rb")
file = pickle.load(pickle_in)
pickle_in.close()


classifiers = [
        # KNeighborsClassifier(),    
        # # NuSVC(probability=True), #NOT USING
        # DecisionTreeClassifier(),
        # # RandomForestClassifier(), #NOT USING
        # AdaBoostClassifier(),
        # # GradientBoostingClassifier(),  #NOT USING
        # GaussianNB(),
        LinearDiscriminantAnalysis(shrinkage="auto"),
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
et_idx = [10,13,14,16,18,19,20,28,30,32,33,34]
fInit = "log_eeg_eco_"


psd_top = [0,  1,  2,  3,  5,  6,  9, 10, 13, 14, 15, 16, 17, 19, 23, 24, 26, 27, 28, 29]

for classifier, parameters in zip(classifiers, tuned_parameters):   
    # Logging for Visual Comparison
    log_cols=["Classifier", "Accuracy", "Log Loss"]
    log = pd.DataFrame(columns=log_cols)

    clf = classifier
    name = clf.__class__.__name__
    
    print()
    print()
    print('Processing with Classifier:', name)
    # if os.path.exists("./output/" +fInit +name +".pickle") == True:
    #     print("File exists:", "./output/" +fInit +name +".pickle")
    #     continue
    
    for i in range(13):
        for j in range(1, 13):
            if j <= i:
                continue
            print("Testing with:", i, j)
            
            X_train         = np.zeros([720, 20])
            y_train         = []
            counter_train   = 0
            
            X_test1         = np.zeros([1*1*2*15, 20])
            X_test2         = np.zeros([1*1*2*15, 20])
            y_test1         = []
            y_test2         = []
            counter_test1   = 0
            counter_test2   = 0
        
            
            
            #### Inter subject classification

            for p in range(13):
                for l in range (2):
                    for s in range (2):
                        for e in range (15):    
                            
                            # if p == i or p == j:    #### Data for testing
                                
                                # temp_chn = []
                                # for c in psd_top :
                                #     temp_chn.append(file[p,l,0,e,c,2048:2663].mean(0))
                                # X_test1[counter_test1, :] = temp_chn
                                # y_test1.append(l)
                                # counter_test1 += 1

                            if p == i:
                                
                                if s ==0: 
                                    temp_chn = []
                                    for c in psd_top :
                                        temp_chn.append(file[p,l,s,e,c,2048:2663].mean(0))                                 
                                    X_train[counter_train, :] = temp_chn                     
                                    y_train.append(l)
                                    counter_train += 1 
                                    
                                else:  
                                    temp_chn = []                                                                       
                                    for c in psd_top :                                        
                                        temp_chn.append(file[p,l,s,e,c,2048:2663].mean(0))
                                    X_test1[counter_test1, :] = temp_chn
                                    y_test1.append(l)
                                    counter_test1 += 1
                                
                            elif p == j:
                                if s ==0:
                                    temp_chn = []
                                    for c in psd_top :
                                        temp_chn.append(file[p,l,s,e,c,2048:2663].mean(0))                                 
                                    X_train[counter_train, :] = temp_chn                     
                                    y_train.append(l)
                                    counter_train += 1 
                                    
                                else:   
                                    temp_chn = []                                                                      
                                    for c in psd_top :                                        
                                        temp_chn.append(file[p,l,s,e,c,2048:2663].mean(0))
                                    X_test2[counter_test2, :] = temp_chn
                                    y_test2.append(l)
                                    counter_test2 += 1
                                
                            else:                   #### Data for training 
                                temp_chn = []
                                for c in psd_top :
                                    temp_chn.append(file[p,l,s,e,c,2048:2663].mean(0))                                 
                                X_train[counter_train, :] = temp_chn                     
                                y_train.append(l)
                                counter_train += 1 
                            
                                
                  
            # X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # X_test1 = np.array(X_test1)
            y_test1 = np.array(y_test1)
            
            # X_test2 = np.array(X_test2)
            y_test2 = np.array(y_test2)    
            
            print(X_train.shape, y_train.shape, X_test1.shape, y_test1.shape, X_test2.shape, y_test2.shape)
                
            scaler  = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test1 = scaler.transform(X_test1)
            X_test2 = scaler.transform(X_test2)
            
            if parameters != {None}:            
                print("# Tuning hyper-parameters for %s" % name)
                clf = GridSearchCV(classifier, parameters, cv=5, scoring=score, n_jobs=-1, verbose=0)
            else:
                clf = GaussianNB()
        
            best_param = None
            clf.fit(X_train, y_train)  
            
            if parameters != {None}:
                best_param = clf.best_params_
                print("Best parameters for:", name)
                print(clf.best_params_)                        
            
            train_predictions = clf.predict(X_test1)
            acc = accuracy_score(y_test1, train_predictions)         
            train_predictions = clf.predict_proba(X_test1)
            ll = log_loss(y_test1, train_predictions)
            print("Accu: {:.2%}".format(acc), end=' ')   
            print("Loss: {:.2%}".format(ll))            
            log_entry = pd.DataFrame([[str(i), acc*100, ll]], columns=log_cols)
            log = log.append(log_entry)         
            
            train_predictions = clf.predict(X_test2)
            acc = accuracy_score(y_test2, train_predictions)    
            train_predictions = clf.predict_proba(X_test2)
            ll = log_loss(y_test2, train_predictions)
            print("Accu: {:.2%}".format(acc), end=' ')        
            print("Loss: {:.2%}".format(ll))            
            log_entry = pd.DataFrame([[str(j), acc*100, ll]], columns=log_cols)
            log = log.append(log_entry)     
                
            print("="*30)
            print("="*30)
            print()
            
            
    pickle_out = open("/home/dcas/l.tilly/Documents/data/lh_erp_inter_eco.pickle", "wb")
    pickle.dump(log, pickle_out)
    pickle_out.close()
                    
    pickle_out = open("./output/" +fInit +name +".pickle", "wb")
    pickle.dump(log, pickle_out)
    pickle_out.close()
        
    val = log.loc[:,['Accuracy']].values
    print(val.min(), val.max(), val.mean(), np.median(val))
