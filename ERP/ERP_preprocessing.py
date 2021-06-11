#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:30:53 2021

@author: l.tilly
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np 
import decimal
import scipy.io
import mne
from mne.preprocessing import ICA
from mne.time_frequency import psd_multitaper

from scipy.interpolate import LinearNDInterpolator

import PIL
from PIL import Image

erp = "/home/dcas/l.tilly/Documents/data/ERP_data_2sec.pickle"
pickle_in = open(erp, "rb")
file = pickle.load(pickle_in)
pickle_in.close()

ch_names = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3',
        'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4',
        'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8',
        'AF4', 'Fp2', 'Fz', 'Cz','REF1','REF2','EOGH','EOGV','ECG1','ECG2']

ch_types = ['eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg',
    'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg',
    'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg',
    'eeg','eeg','stim','stim','eog','eog','ecg','ecg']

file_erp = file[:,:,:,:,:,:].copy()
newSFreq = float(50.0)
sfreq = float(2048.0)
error_array = []

def CreatingMneRawObject(df, ch_names, ch_types, sfreq):        
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(df, info)    
    return raw


def bdpass_filtering(epoch, ch_names, ch_types, sfreq, minFreq=1.0, maxFreq=30.0, fmax=50):
    raw = None
    raw = CreatingMneRawObject(epoch, ch_names, ch_types, sfreq)

    filtered_data = raw.filter(minFreq, maxFreq, n_jobs=1)
    return filtered_data.get_data()


def Bad_chn(epoch):                 
    bad_chn = []    
    mean_array = epoch.mean(1)
    mean_all = mean_array.mean(0)
    std_all = mean_array.std(0)

    # bad_chn detection 
    for c in range (len(mean_array)): 
        if abs(mean_array[c]) > abs(mean_all + 2*std_all) :   
            bad_chn.append(ch_names[c])
    print(bad_chn)          
    return bad_chn

array = np.zeros([13,2,2,15,38,100])

for p in range(13):
    for l in range (2):
        for s in range (2):
            for e in range (15):
                
############################# BAND-PASS FILTERING #############################

                epoch = np.array(file[p,l,s,e,:,:])
                file_erp[p,l,s,e,:,:] = bdpass_filtering(epoch, ch_names, ch_types, sfreq)
                
                
# ############################### INTERPOLATION #################################

                epoch = np.array(file_erp[p,l,s,e,:32,:]) 
                epoch = CreatingMneRawObject(epoch, ch_names[:32], ch_types[:32], sfreq)
                epoch.info['bads'] = Bad_chn(epoch.get_data())
                epoch.set_montage("biosemi32")
                epoch.  (reset_bads=True, verbose=False) # Interpolation step
                
                
# ############################### RE-REFERENCING #################################
                
                epoch.set_eeg_reference('average', projection=True)
                epoch.apply_proj()  
                
                epoch.get_data()[~np.isfinite(epoch.get_data())] = 0 # check inf ou NaN 
              
                file_erp[p,l,s,e,:32,:] = epoch.get_data() 
        
#################################### ICA ########################################

                try: 

                    epoch = CreatingMneRawObject(file_erp[p,l,s,e,:,:], ch_names, ch_types, sfreq)
                    epochCopy = epoch.copy()
                    epochCopy.load_data()
                    ica = ICA(random_state=97)
                    ica.fit(epochCopy)
                    epoch.load_data()
                    
                    # find which ICs match the EOG pattern OCCULAR CORRECTION
                    eog_indices, eog_scores = ica.find_bads_eog(epoch, threshold=2.6)
                    ica.exclude = eog_indices      
                    if(len(ica.exclude)>0):
                        ica.apply(epochCopy)
                    else:
                        epochCopy = epoch.copy()
                    
############################### BASELINE CORRECTION #############################
                    
                    file_erp[p,l,s,e,:,:] = epochCopy.get_data()    
                    first = True
                    for c in range (32):
                        before = 0
                        for d in range (1024,1434):
                            before = before + file_erp[p,l,s,e,c,d]  
                        avg = before/(1434-1024)
                        for i in range (0,4096):
                            file_erp[p,l,s,e,c,i] = file_erp[p,l,s,e,c,i] - avg

                except:
                    
                    ee = str(p) + "," + str(l) + "," + str(s) + "," + str(e) 
                    error_array.append(ee)
            
                raw = CreatingMneRawObject(file_erp[p,l,s,e,:,:], ch_names, ch_types, sfreq)

############################### RESAMPLING DATA ###############################
                raw = raw.resample(newSFreq, npad='auto')
                
                array[p,l,s,e,:,:] = raw.get_data() 
                
                
                
                

pickle_out = open("/home/dcas/l.tilly/Documents/data/ERP_data_2sec_processed_resampled_50.pickle", "wb")
pickle.dump(array, pickle_out)
pickle_out.close()

processed = CreatingMneRawObject(file_erp[0,0,0,0,:32,:], ch_names[:32], ch_types[:32], sfreq)
processed.plot(n_channels=32, scalings=25, title='Auto-scaled Data from arrays',show=True, block=True)

# raw = CreatingMneRawObject(file[0,0,0,0,:32,:], ch_names[:32], ch_types[:32], sfreq)
# raw.plot(n_channels=32, scalings=25, title='Auto-scaled Data from arrays',show=True, block=True)

# bd_pass = CreatingMneRawObject(file[0,0,0,0,:32,:], ch_names[:32], ch_types[:32], sfreq)
# bd_pass.plot(n_channels=32, scalings=25, title='Auto-scaled Data from arrays',show=True, block=True)

# int = CreatingMneRawObject(file_erp[0,0,0,0,:32,:], ch_names[:32], ch_types[:32], sfreq)
# int.plot(n_channels=32, scalings=25, title='Auto-scaled Data from arrays',show=True, block=True)

# bd_pass = bdpass_filtering(file_erp[3,0,0,0,:32,:], ch_names[:32], ch_types[:32], sfreq, 1,30,50)
# bd_pass.plot(n_channels=32, scalings=25, title='Auto-scaled Data from arrays',show=True, block=True)

# for p in range(13):
#     for l in range (2):
#         for s in range (2):
#             for e in range (15):
#                 file_erp[p,l,s,e,:,:] = bdpass_filtering(file_erp[p,l,s,e,:,:], ch_names[:32], ch_types[:32], sfreq, minFreq=1.0, maxFreq=30.0, fmax=50)

# pickle_out = open("/home/dcas/l.tilly/Documents/data/ERP_data_2sec_bdpass.pickle", "wb")
# pickle.dump(file_erp, pickle_out)
# pickle_out.close()

#################### channel rejection ############## obtention bad_chn ######

# def Bad_chn(epoch):                 
#     bad_chn = []    
#     mean_array = epoch.mean(1)
#     mean_all = mean_array.mean(0)
#     std_all = mean_array.std(0)

#     # bad_chn detection 
#     for c in range (len(mean_array)): 
#         if abs(mean_array[c]) > abs(mean_all + 2*std_all) :   
#             bad_chn.append(c)
            
#     # making the bad_chn = 0      
#     for i in bad_chn : 
#         epoch[i,:4096] = 0 
        
#     # the averaging of the chn left 
#     mean_ref = epoch.sum(0)/(32-len(bad_chn))

#     # replace all the bad_chn by average of good 
#     for i in bad_chn : 
#         epoch[i,:4096] = mean_ref
                
#     return epoch


# def ref(epoch):
#      raw = CreatingMneRawObject(file_erp[p,l,s,e,:,:], ch_names[:32], ch_types[:32], sfreq)
#      rawCopy = raw.copy()
#      rawCopy.set_eeg_reference('average', projection=True)
#      rawCopy.apply_proj() 
     
#      return rawCopy.get_data()


# for p in range(13):
#     for l in range (2):
#         for s in range (2):
#             for e in range (15):
#                 file_erp[p,l,s,e,:,:] = Bad_chn(file_erp[p,l,s,e,:,:])   

# pickle_out = open("/home/dcas/l.tilly/Documents/data/ERP_data_2sec_bdpass_interpolated.pickle", "wb")
# pickle.dump(file_erp, pickle_out)
# pickle_out.close()


# re-referencing 

# for p in range(13):
#     for l in range (2):
#         for s in range (2):
#             for e in range (15):
#                file_erp[p,l,s,e,:,:] = ref(file_erp[p,l,s,e,:,:])
               
# pickle_out = open("/home/dcas/l.tilly/Documents/data/ERP_data_2sec_bdpass_int_ref.pickle", "wb")
# pickle.dump(file_erp, pickle_out)
# pickle_out.close()   


# if len(ica.exclude) > 0:
#     print(e, ica.exclude)
#     print(bad_chn)
 
#                 epochCopy = epoch.copy()
#                 epochCopy.load_data()
#                 ica = ICA(random_state=97)
#                 ica.fit(epochCopy)
#                 epoch.load_data()
                
#                 # find which ICs match the EOG pattern OCCULAR CORRECTION
#                 eog_indices, eog_scores = ica.find_bads_eog(epoch, threshold=2.6)
#                 ica.exclude = eog_indices      
#                 if(len(ica.exclude)>0):
#                     ica.apply(epochCopy)
#                 else:
#                     epochCopy = epoch.copy()

#                 #epochCopy = epochCopy.get_data()
                
#                 erp_preprocessed[p,l,s,e,:,:] = epochCopy.get_data()
                
#                 # if len(ica.exclude) > 0:
#                 #     print(e, ica.exclude)
#                 #     print(bad_chn)

##################### see graphs individually #################################
# raw = CreatingMneRawObject(file_erp[0,0,0,0,:32,:], ch_names[:32], ch_types[:32], sfreq)
# raw.plot(n_channels=32, scalings=25, title='Auto-scaled Data from arrays',show=True, block=True)


# pickle_out = open("/home/dcas/l.tilly/Documents/data/ERP_data_2sc_bdpass_interpolated.pickle", "wb")
# pickle.dump(file_erp, pickle_out)
# pickle_out.close()



              
######################## AVERAGE RE-REFERENCING ############################## 


 

########################### OCCULAR CORRECTION ############################### 


             
#                 ###### ICA #######
                
          
          
# print("#################")
# print("Plotting graphs")
# print("#################")

# # données pre-processed filtrées 
# print("No correction")
# raw = CreatingMneRawObject(file[p,l,s,e,:,:], ch_names, ch_types, sfreq)
# raw.filter(minFreq, maxFreq, n_jobs=1)
# plt1 = raw.plot(n_channels=38, scalings=25, title='ERPs participant' + str(p)+ 'with filtered data',show=True, block=True)
# plt1.savefig("1.jpg")

# print("#################")
# print("Interpolation")
# # données preprocessed filtrées + interpolation après détection bad chn 
# raw = CreatingMneRawObject(epoch.get_data(), ch_names, ch_types, sfreq)
# plt2 = raw.plot(n_channels=38, scalings=25, title='ERPs participant' + str(p)+ 'with filtration and interpolation',show=True, block=True)
# plt2.savefig("2.jpg")

# print("#################")
# print("Interpolation + ICA")
# # données preprocessed filtrées + interpolation après détection bad chn + ICA
# raw = CreatingMneRawObject(epochCopy.get_data(),  ch_names, ch_types, sfreq)
# plt3 = raw.plot(n_channels=38, scalings=25, title='ERPs participant' + str(p)+ 'with filtration, interpolation and ICA',show=True, block=True)
# plt3.savefig("3.jpg")


# list_im = ["1.jpg", "2.jpg", "3.jpg"]
# imgs    = [ PIL.Image.open(i) for i in list_im ]
# # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
# min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
# imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
# imgs_comb = PIL.Image.fromarray( imgs_comb)
# imgs_comb.save( './images_preprocessing2.6/p'+str(p) +'l'+str(l) +'s'+str(s) +'e'+str(e) +'bad'+str(len(bad_chn)) +'ica'+str(len(ica.exclude))  +'.jpg' ) 


# mean_erp_low = erp_preprocessed[p,0,s,:,:,:].mean(0)
# mean_erp_high = erp_preprocessed[p,1,s,:,:,:].mean(0)

# std_erp_low = erp_preprocessed[p,0,s,:,12,:].std(0)
# std_erp_high = erp_preprocessed[p,1,s,:,12,:].std(0)


# def drange(x, y, jump):
#   while x < y:
#     yield float(x)
#     x += decimal.Decimal(jump)
    
# time = list(drange(-200,800,'0.48828125'))

# f, ax = plt.subplots(figsize=(10,7))

# ax.plot(time, mean_erp_low, color='g', label="Low WL")
# ax.plot(time, mean_erp_high, color='r', label="High WL")

# # ax.fill_between(time, mean_erp_low+std_erp_low, mean_erp_low-std_erp_low, facecolor='g', alpha=0.3)
# # ax.fill_between(time, mean_erp_high+std_erp_high, mean_erp_high-std_erp_high, facecolor='r', alpha=0.3)
# # # ax.plot(time, mean_erp_low, color='g', label="Low WL")
# # # ax.plot(time, mean_erp_high, color='r', label="High WL")

# # ax.set(title=' ERPs_correction average for all participants', xlabel='Time (ms)',ylabel='Potential (microV)')
# # ax.legend()
# # plt.grid()
# # plt.show()


# # "%% MATLAB %%"

# # # pickle_out = open("./output/erp_preprocessed_participant5.pickle", "wb")
# # # pickle.dump(erp_preprocessed_participant, pickle_out)
# # # pickle_out.close()

# # # final_data=pickle.load( open( "./output/erp_preprocessed_participant5.pickle", "rb" ) )
# # # scipy.io.savemat('./output/erp_preprocessed_participant5.mat', mdict={'erp_preprocessed_p5': final_data})
