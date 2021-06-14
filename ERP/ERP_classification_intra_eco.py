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
        
def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn

#### Load ERP data
erp = "/home/dcas/l.tilly/Documents/data/ERP_data_2sec_processed.pickle"
pickle_in = open(erp, "rb")
file = pickle.load(pickle_in)
pickle_in.close()



# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)


classifiers = [
#         KNeighborsClassifier(),    
# #        NuSVC(probability=True),
#         DecisionTreeClassifier(),
# #        RandomForestClassifier(),
#         AdaBoostClassifier(),
# #        GradientBoostingClassifier(),
#         GaussianNB(),
        LinearDiscriminantAnalysis(shrinkage="auto"),
#         QuadraticDiscriminantAnalysis(),
#         SVC()
    ]

tuned_parameters = [
#         {'n_neighbors':[2, 3, 4, 5, 6, 7, 8, 9, 10], 'weights' : ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
# #        {'kernel': ['linear', 'poly', 'rbf'], 'gamma': [0.1,0.15,0.2,0.5,1,1.5]},
#         {'max_depth': [ 5, 10, 15, 20, 25] },
# #        {'max_depth': [ 5, 10, 15], 'n_estimators': [1, 3, 5, 7, 9], 'max_features': [1, 2, 3, 4, 5] },
#         {'n_estimators': [20, 30, 40, 50, 60, 70, 80] },
# #        {"loss":["deviance"], "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],"min_samples_split": np.linspace(0.1, 0.5, 12), "min_samples_leaf": np.linspace(0.1, 0.5, 12),"max_depth":[3,5,8],"max_features":["log2","sqrt"],"criterion": ["friedman_mse",  "mae"],"subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0]},
#         {None},
        {'solver': ["lsqr","eigen"]},
#         {'reg_param' : [0.0, 1.0, 2.0, 3.0]},
#         {'kernel': ['linear', 'poly', 'rbf'], 'gamma': [1e-1, 1.5e-1, 2e-1],'C': [0.5, 1, 2] }
    ]
        
score = 'accuracy'

for classifier, parameters in zip(classifiers, tuned_parameters):    
    clf = classifier
    name = clf.__class__.__name__
    
    # print()
    # print()
    # print('Processing with Classifier:', name)
    # if os.path.exists("./output/log_eeg_intra_" +name +".pickle") == True:
    #     continue
    
    for i in range(13):
        print("#############################################")
        print("Testing with participant:", i)
             
        X_train         = np.zeros([1*1*2*15, 20])
        y_train         = []
        counter_train   = 0
        
        X_test         = np.zeros([1*1*2*15, 20])
        y_test         = []
        counter_test   = 0
        psd_top = [0,  1,  2,  3,  5,  6,  9, 10, 13, 14, 15, 16, 17, 19, 23, 24, 26, 27, 28, 29]

        #### Intra subject classification

        for l in range(2):            
            for p in range(13):
                for e in range(15):                        
                    if p == i:
                        erp = []
                        for c in psd_top :
                            erp.append(file[p,l,0,e,c,2048:2663].mean(0))
                        X_train[counter_train, :] = erp
                        y_train.append(l)
                        counter_train += 1    
                        
                        erp = []
                        for c in psd_top :
                            erp.append(file[p,l,1,e,c,2048:2663].mean(0))
                        X_test[counter_test, :] = erp
                        y_test.append(l)
                        counter_test += 1
    
                
        y_train = np.array(y_train)
        y_test = np.array(y_test)            
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
            
        if name == "KNeighborsClassifier":
            clf = KNeighborsClassifier(n_neighbors=best_param['n_neighbors'], weights=best_param['weights'], algorithm=best_param['algorithm'])
        elif name == "NuSVC":
            clf = NuSVC(kernel='poly', gamma=1, probability=True)
        elif name == "DecisionTreeClassifier":
            clf = DecisionTreeClassifier(max_depth=best_param['max_depth'])
        elif name == "RandomForestClassifier":
            clf = RandomForestClassifier(max_depth=best_param['max_depth'], n_estimators=best_param['n_estimators'], max_features=best_param['max_features'])
        elif name == "AdaBoostClassifier":
            clf = AdaBoostClassifier(n_estimators=best_param['n_estimators'])
        elif name == "GradientBoostingClassifier":
            clf = GradientBoostingClassifier(loss=best_param['loss'], learning_rate=best_param['learning_rate'], min_samples_split=best_param['min_samples_split'], min_samples_leaf=best_param['min_samples_leaf'], max_depth=best_param['max_depth'], max_features=best_param['max_features'], criterion=best_param['criterion'], subsample=best_param['subsample'])
        elif name == "GaussianNB":
            clf = GaussianNB()
        elif name == "LinearDiscriminantAnalysis":
            clf = LinearDiscriminantAnalysis(shrinkage="auto", solver=best_param['solver'])
        elif name == "QuadraticDiscriminantAnalysis":
            clf = QuadraticDiscriminantAnalysis(reg_param=best_param['reg_param'])
        elif name == "SVC":
            clf = SVC(kernel=best_param['kernel'], C=best_param['C'], gamma=best_param['gamma'], probability=True)                
        
        lw_train = X_train[:15,:]
        hw_train = X_train[15:,:]
        print(lw_train.shape, hw_train.shape)
        lw_train = [lw_train[:10,:], lw_train[5:,:], np.concatenate((lw_train[:5,:], lw_train[10:,:]), axis=0)]
        hw_train = [hw_train[:10,:], hw_train[5:,:], np.concatenate((hw_train[:5,:], hw_train[10:,:]), axis=0)]
        
        lw_test = X_test[:15,:]
        hw_test = X_test[15:,:]
        print(lw_test.shape, hw_test.shape)
        lw_test = [lw_test[:10,:], lw_test[5:,:], np.concatenate((lw_test[:5,:], lw_test[10:,:]), axis=0)]
        hw_test = [hw_test[:10,:], hw_test[5:,:], np.concatenate((hw_test[:5,:], hw_test[10:,:]), axis=0)]
        
        for lt in range(len(lw_train)):
            for ht in range(len(hw_train)):
                X_train_g = np.concatenate((lw_train[lt], hw_train[ht]), axis=0)
                y_train_g = np.concatenate((y_train[:10], y_train[20:]), axis=0)                                
                
                clf.fit(X_train_g, y_train_g)                  
                
                for ltst in range(len(lw_test)):
                    for htst in range(len(hw_test)):
                        X_test_g = np.concatenate((lw_test[ltst], hw_test[htst]), axis=0)
                        y_test_g = np.concatenate((y_test[:10], y_test[20:]), axis=0)
                        
                        train_predictions = clf.predict(X_test_g)
                        acc = accuracy_score(y_test_g, train_predictions)
                        print("Accu: {:.2%}".format(acc), end=' ')
                        
                        train_predictions = clf.predict_proba(X_test_g)
                        ll = log_loss(y_test_g, train_predictions)
                        print("Loss: {:.2%}".format(ll))
                        
                        log_entry = pd.DataFrame([[i, acc*100, ll]], columns=log_cols)
                        log = log.append(log_entry)            
            
        print("="*30)
        print("="*30)
        print()
    
    pickle_out = open("/home/dcas/l.tilly/Documents/data/lh_erp_intra_eco.pickle", "wb")
    pickle.dump(log, pickle_out)
    pickle_out.close()

        
    val = log.loc[:,['Accuracy']].values
    print(val.min(), val.max(), val.mean(), np.median(val))






#val = val[:330]
#val = np.reshape(val, [-1, 10])
#print(val.shape)
#val = val.mean(0)
#print(val)


#
#nam = log.loc[:,['Classifier']].values
#print(nam.shape)
#nam = nam[:9]
#nam = np.reshape(nam, [-1])
#print(nam)
##### Classification output for EEG
##nam = np.array(['KNeighborsClassifier', 'NuSVC', 'DecisionTreeClassifier',
##       'RandomForestClassifier', 'AdaBoostClassifier',
##       'GradientBoostingClassifier', 'GaussianNB',
##       'LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis',
##       'SVC'], dtype=object)
#
#all_log_cols=["Classifier", "Accuracy"]
#all_log = pd.DataFrame(columns=all_log_cols)
#all_log = all_log.append(pd.DataFrame([[nam, val]], columns=all_log_cols))
#
#sns.set_color_codes("muted")
#sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
#
#plt.xlabel('Accuracy %')
#plt.title('Classifier Accuracy')
#plt.show()
#
