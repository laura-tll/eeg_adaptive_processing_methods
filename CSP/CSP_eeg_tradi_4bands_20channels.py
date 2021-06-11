#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 22:51:39 2021

@author: l.tilly
"""




import scipy as scp
from scipy import signal
import numpy as np
import pickle
import pandas as pd

import mne
from mne.decoding import CSP

# from sklearn.pipeline import make_pipeline
# from sklearn.svm import SVC
# from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, log_loss

from scipy.signal import butter, lfilter



#### Load EEG data
eeg = "../data/lh_EEG_Signal.pickle"
pickle_in = open(eeg, "rb")
lh_eeg = pickle.load(pickle_in)
pickle_in.close()



newSFreq = float(512.0)
##### Variables for EEG processing ####
ch_names = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3',
        'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4',
        'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8',
        'AF4', 'Fp2', 'Fz', 'Cz']
ch_types = ['eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg',
    'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg',
    'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg',
    'eeg','eeg']


def CreatingMneRawObject(data, ch_names, ch_types, sfreq):        
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)    
    return raw


##### Load ECG data
#ecg = "../data/features/lh_ECG_normalized.pickle"
#pickle_in = open(ecg, "rb")
#lh_ecg = pickle.load(pickle_in)
#pickle_in.close()
#
##### Load ET data
#et = "../data/features/lh_ET.pickle"
#pickle_in = open(et, "rb")
#lh_et = pickle.load(pickle_in)

# Logging for Visual Comparison
log_cols=["Participants", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# def check_accuracy(test_dict, csp):    
#     correct = 0
#     epochs = 62 # test_dict[0].shape[0]
#     for i in range(62):
#         for key in test_dict.keys():
#             a = test_dict[key][i, :, :]
#             y = CSP_clf.evaluate(a)
#             if key == 0 and y == -1:
#                 correct += 1
#             elif key == 1 and y == 1:
#                 correct += 1
    
#     return (correct / (epochs * 2))


 
    
for i in range(13):
    for j in range(1, 13):
        if j <= i:
            continue
        print("Testing with:", i, j)
        X_train = np.array([])
        y_train = []
        X_test1 = np.array([])   
        y_test1 = []
        X_test2 = np.array([])   
        y_test2 = []
        first_time_train = True
        first_time_test1 = True
        first_time_test2 = True
        
        psd_top = [0,  1,  2,  3,  5,  6,  9, 10, 13, 14, 15, 16, 17, 19, 23, 24, 26, 27, 28, 29]
        
        #### Inter subject classification
        participants, loads, subLoads, epochs, channels, features = lh_eeg.shape
        for l in range(loads):            
            for p in range(participants):
                for e in range(0, epochs, 1):         
                    X_train_tmp = np.array([])
                    X_test_tmp = np.array([])
                    first_filt = True                       
                    
                    if p == i:    #### Data for testing                        
                        for c in psd_top: #channels only first 32 which are eeg channels                                                                                                  
                            lte_filt1 = butter_bandpass_filter(lh_eeg[p,l,0,e,c,:], 4, 45, 512, order=6)
                            lte_filt1 = np.expand_dims(np.expand_dims(lte_filt1, axis=0), axis=0)
                            
                            lte_filt2 = butter_bandpass_filter(lh_eeg[p,l,1,e,c,:], 4, 45, 512, order=6)
                            lte_filt2 = np.expand_dims(np.expand_dims(lte_filt2, axis=0), axis=0)
                            
                            lte_filt  = np.concatenate((lte_filt1, lte_filt2), axis=0)
                    
                            if first_filt:
                                X_test_tmp   = lte_filt
                                first_filt   = False
                            else:
                                X_test_tmp   = np.concatenate((X_test_tmp, lte_filt), axis=1)

                        if first_time_test1:                                                        
                            X_test1              = X_test_tmp
                            first_time_test1     = False
                        else:
                            X_test1     = np.concatenate((X_test1, X_test_tmp), axis=0)
                        
                        y_test1.append(l)
                        y_test1.append(l)
                            
                    elif p == j:    #### Data for testing                        
                        for c in psd_top: #channels only first 32 which are eeg channels                                  
                            lte_filt1 = butter_bandpass_filter(lh_eeg[p,l,0,e,c,:], 4, 45, 512, order=6)
                            lte_filt1 = np.expand_dims(np.expand_dims(lte_filt1, axis=0), axis=0)
                            
                            lte_filt2 = butter_bandpass_filter(lh_eeg[p,l,1,e,c,:], 4, 45, 512, order=6)
                            lte_filt2 = np.expand_dims(np.expand_dims(lte_filt2, axis=0), axis=0)
                            
                            lte_filt  = np.concatenate((lte_filt1, lte_filt2), axis=0)
                                            
                            if first_filt:
                                X_test_tmp  = lte_filt
                                first_filt  = False
                            else:
                                X_test_tmp  = np.concatenate((X_test_tmp, lte_filt), axis=1)

                        if first_time_test2:                                                        
                            X_test2     = X_test_tmp
                            first_time_test2  = False
                        else:
                            X_test2     = np.concatenate((X_test2, X_test_tmp), axis=0)
                                                
                        y_test2.append(l)
                        y_test2.append(l)
                            
                    else:                     
                        for con in range(2):
                            X_train_tmp = np.array([])
                            X_test_tmp = np.array([])
                            first_filt = True        
                            for c in psd_top: #channels only first 32 which are eeg channels                                                        
                                ltr_filt = butter_bandpass_filter(lh_eeg[p,l,con,e,c,:], 4, 45, 512, order=6)                        
                                ltr_filt = np.expand_dims(np.expand_dims(ltr_filt, axis=0), axis=0)
                        
                                if first_filt:
                                    X_train_tmp = ltr_filt                                    
                                    first_filt  = False
                                else:
                                    X_train_tmp = np.concatenate((X_train_tmp, ltr_filt), axis=1)

                            if first_time_train:
                                X_train           = X_train_tmp
                                first_time_train  = False
                            else:
                                X_train     = np.concatenate((X_train, X_train_tmp), axis=0)   
                        
                            y_train.append(l)                              
                    
                                      
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        X_test1 = np.array(X_test1)
        y_test1 = np.array(y_test1)
        
        X_test2 = np.array(X_test2)
        y_test2 = np.array(y_test2)    
        
        print(X_train.shape, y_train.shape, X_test1.shape, y_test1.shape, X_test2.shape, y_test2.shape)
        
        lda             = LinearDiscriminantAnalysis()
        csp             = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        eeg_train_tmp   = csp.fit_transform(X_train, y_train)       
        lda.fit(eeg_train_tmp, y_train)
        
        X_test1             = csp.transform(X_test1)        
        train_predictions   = lda.predict(X_test1)
        acc = accuracy_score(y_test1, train_predictions)         
        train_predictions   = lda.predict_proba(X_test1)
        ll = log_loss(y_test1, train_predictions)
        print("Accu 1: {:.2%}".format(acc), end=' ')   
        print("Loss: {:.2%}".format(ll))            
        log_entry = pd.DataFrame([[str(i), acc*100, ll]], columns=log_cols)
        log = log.append(log_entry)         
        
        X_test2             = csp.transform(X_test2)
        train_predictions   = lda.predict(X_test2)
        acc = accuracy_score(y_test2, train_predictions)    
        train_predictions   = lda.predict_proba(X_test2)
        ll = log_loss(y_test2, train_predictions)
        print("Accu 2: {:.2%}".format(acc), end=' ')        
        print("Loss: {:.2%}".format(ll))            
        log_entry = pd.DataFrame([[str(j), acc*100, ll]], columns=log_cols)
        log = log.append(log_entry)     
            
        print("="*30)
        print("="*30)
        print()
                
pickle_out = open("./output/log_eeg_tradi_csp_4bands_20channels.pickle", "wb")
pickle.dump(log, pickle_out)
pickle_out.close()
    
val = log.loc[:,['Accuracy']].values
print(val.min(), val.max(), val.mean(), np.median(val))
                   



# pickle_in = open("./output/log_eeg_intra_csp.pickle", "rb")
# log = pickle.load(pickle_in)
# pickle_in.close()
