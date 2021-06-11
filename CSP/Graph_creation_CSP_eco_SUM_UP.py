#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 12:12:10 2021

@author: l.tilly
"""

import pickle
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

############# NO CSP ##############

file = "./output/log_eeg_eco_theta_LinearDiscriminantAnalysis"
pickle_in = open(file +".pickle", "rb")
no_csp_theta_20c = pickle.load(pickle_in)
no_csp_theta_20c['name'] = 'No_CSP_20c'
no_csp_theta_20c['band'] = 'theta'
no_csp_theta_20c = no_csp_theta_20c.rename(columns={"Classifier": "Participants"})
pickle_in.close()  

file = "./output/log_eeg_eco_theta_32channels_LinearDiscriminantAnalysis"
pickle_in = open(file +".pickle", "rb")
no_csp_theta_32c = pickle.load(pickle_in)
no_csp_theta_32c['name'] = 'No_CSP_32c'
no_csp_theta_32c['band'] = 'theta'
no_csp_theta_32c = no_csp_theta_32c.rename(columns={"Classifier": "Participants"})
pickle_in.close()  




file = "./output/log_eeg_eco_alpha_LinearDiscriminantAnalysis"
pickle_in = open(file +".pickle", "rb")
no_csp_alpha_20c = pickle.load(pickle_in)
no_csp_alpha_20c['name'] = 'No_CSP_20c'
no_csp_alpha_20c['band'] = 'alpha'
no_csp_alpha_20c = no_csp_alpha_20c.rename(columns={"Classifier": "Participants"})
pickle_in.close()  

file = "./output/log_eeg_eco_alpha_32channels_LinearDiscriminantAnalysis"
pickle_in = open(file +".pickle", "rb")
no_csp_alpha_32c = pickle.load(pickle_in)
no_csp_alpha_32c['name'] = 'No_CSP_32c'
no_csp_alpha_32c['band'] = 'alpha'
no_csp_alpha_32c = no_csp_alpha_32c.rename(columns={"Classifier": "Participants"})
pickle_in.close()  





file = "./output/log_eeg_eco_beta_LinearDiscriminantAnalysis"
pickle_in = open(file +".pickle", "rb")
no_csp_beta_20c = pickle.load(pickle_in)
no_csp_beta_20c['name'] = 'No_CSP_20c'
no_csp_beta_20c['band'] = 'beta'
no_csp_betaa_20c = no_csp_beta_20c.rename(columns={"Classifier": "Participants"})
pickle_in.close()  

file = "./output/log_eeg_eco_beta_32channels_LinearDiscriminantAnalysis"
pickle_in = open(file +".pickle", "rb")
no_csp_beta_32c = pickle.load(pickle_in)
no_csp_beta_32c['name'] = 'No_CSP_32c'
no_csp_beta_32c['band'] = 'beta'
no_csp_beta_32c = no_csp_beta_32c.rename(columns={"Classifier": "Participants"})
pickle_in.close()  






file = "./output/log_eeg_eco_gamma_LinearDiscriminantAnalysis"
pickle_in = open(file +".pickle", "rb")
no_csp_gamma_20c = pickle.load(pickle_in)
no_csp_gamma_20c['name'] = 'No_CSP_20c'
no_csp_gamma_20c['band'] = 'gamma'
no_csp_gamma_20c = no_csp_gamma_20c.rename(columns={"Classifier": "Participants"})
pickle_in.close()  

file = "./output/log_eeg_eco_gamma_32channels_LinearDiscriminantAnalysis"
pickle_in = open(file +".pickle", "rb")
no_csp_gamma_32c = pickle.load(pickle_in)
no_csp_gamma_32c['name'] = 'No_CSP_32c'
no_csp_gamma_32c['band'] = 'gamma'
no_csp_gamma_32c = no_csp_gamma_32c.rename(columns={"Classifier": "Participants"})
pickle_in.close()  





file = "./output/log_eeg_eco_4bands_LinearDiscriminantAnalysis"
pickle_in = open(file +".pickle", "rb")
no_csp_4bands_20c = pickle.load(pickle_in)
no_csp_4bands_20c['name'] = 'No_CSP_20c'
no_csp_4bands_20c['band'] = '4bands'
no_csp_4bands_20c = no_csp_4bands_20c.rename(columns={"Classifier": "Participants"})
pickle_in.close()  

file = "./output/log_eeg_eco_4bands_32channels_LinearDiscriminantAnalysis"
pickle_in = open(file +".pickle", "rb")
no_csp_4bands_32c = pickle.load(pickle_in)
no_csp_4bands_32c['name'] = 'No_CSP_32c'
no_csp_4bands_32c['band'] = '4bands'
no_csp_4bands_32c = no_csp_4bands_32c.rename(columns={"Classifier": "Participants"})
pickle_in.close() 




############ CSP4 and CSP6 #####################

# THETA 


file = "./output/log_eeg_csp4_inter_eco_theta_20channels"
pickle_in = open(file +".pickle", "rb")
csp4_theta_20c = pickle.load(pickle_in)
csp4_theta_20c['name'] = 'CSP_20c_4f'
csp4_theta_20c['band'] = 'theta'
pickle_in.close() 

file = "./output/log_eeg_csp6_inter_eco_theta_20channels"
pickle_in = open(file +".pickle", "rb")
csp6_theta_20c = pickle.load(pickle_in)
csp6_theta_20c['name'] = 'CSP_20c_6f'
csp6_theta_20c['band'] = 'theta'
pickle_in.close()   


file = "./output/log_eeg_csp_inter_theta"
pickle_in = open(file +".pickle", "rb")
csp4_theta_32c = pickle.load(pickle_in)
csp4_theta_32c['name'] = 'CSP_32c_4f'
csp4_theta_32c['band'] = 'theta'
pickle_in.close() 

file = "./output/log_eeg_eco_csp6_theta"
pickle_in = open(file +".pickle", "rb")
csp6_theta_32c = pickle.load(pickle_in)
csp6_theta_32c['name'] = 'CSP_32c_6f'
csp6_theta_32c['band'] = 'theta'
pickle_in.close()


# ALPHA 

file = "./output/log_eeg_csp4_inter_eco_alpha_20channels"
pickle_in = open(file +".pickle", "rb")
csp4_alpha_20c = pickle.load(pickle_in)
csp4_alpha_20c['name'] = 'CSP_20c_4f'
csp4_alpha_20c['band'] = 'alpha'
pickle_in.close() 

file = "./output/log_eeg_csp6_inter_eco_alpha_20channels"
pickle_in = open(file +".pickle", "rb")
csp6_alpha_20c = pickle.load(pickle_in)
csp6_alpha_20c['name'] = 'CSP_20c_6f'
csp6_alpha_20c['band'] = 'alpha'
pickle_in.close()  

file = "./output/log_eeg_csp_inter_alpha"
pickle_in = open(file +".pickle", "rb")
csp4_alpha_32c = pickle.load(pickle_in)
csp4_alpha_32c['name'] = 'CSP_32c_4f'
csp4_alpha_32c['band'] = 'alpha'
pickle_in.close() 

file = "./output/log_eeg_eco_csp6_alpha"
pickle_in = open(file +".pickle", "rb")
csp6_alpha_32c = pickle.load(pickle_in)
csp6_alpha_32c['name'] = 'CSP_32c_6f'
csp6_alpha_32c['band'] = 'alpha'
pickle_in.close()


# BETA 

file = "./output/log_eeg_csp4_inter_eco_beta_20channels"
pickle_in = open(file +".pickle", "rb")
csp4_beta_20c = pickle.load(pickle_in)
csp4_beta_20c['name'] = 'CSP_20c_4f'
csp4_beta_20c['band'] = 'beta'
pickle_in.close() 

file = "./output/log_eeg_csp6_inter_eco_beta_20channels"
pickle_in = open(file +".pickle", "rb")
csp6_beta_20c = pickle.load(pickle_in)
csp6_beta_20c['name'] = 'CSP_20c_6f'
csp6_beta_20c['band'] = 'beta'
pickle_in.close()  

file = "./output/log_eeg_csp_inter_beta"
pickle_in = open(file +".pickle", "rb")
csp4_beta_32c = pickle.load(pickle_in)
csp4_beta_32c['name'] = 'CSP_32c_4f'
csp4_beta_32c['band'] = 'beta'
pickle_in.close() 

file = "./output/log_eeg_eco_csp6_beta"
pickle_in = open(file +".pickle", "rb")
csp6_beta_32c = pickle.load(pickle_in)
csp6_beta_32c['name'] = 'CSP_32c_6f'
csp6_beta_32c['band'] = 'beta'
pickle_in.close()

# GAMMA 

file = "./output/log_eeg_csp4_inter_eco_gamma_20channels"
pickle_in = open(file +".pickle", "rb")
csp4_gamma_20c = pickle.load(pickle_in)
csp4_gamma_20c['name'] = 'CSP_20c_4f'
csp4_gamma_20c['band'] = 'gamma'
pickle_in.close() 

file = "./output/log_eeg_csp6_inter_eco_gamma_20channels"
pickle_in = open(file +".pickle", "rb")
csp6_gamma_20c = pickle.load(pickle_in)
csp6_gamma_20c['name'] = 'CSP_20c_6f'
csp6_gamma_20c['band'] = 'gamma'
pickle_in.close()  

file = "./output/log_eeg_csp_inter_gamma"
pickle_in = open(file +".pickle", "rb")
csp4_gamma_32c = pickle.load(pickle_in)
csp4_gamma_32c['name'] = 'CSP_32c_4f'
csp4_gamma_32c['band'] = 'gamma'
pickle_in.close() 

file = "./output/log_eeg_eco_csp6_gamma"
pickle_in = open(file +".pickle", "rb")
csp6_gamma_32c = pickle.load(pickle_in)
csp6_gamma_32c['name'] = 'CSP_32c_6f'
csp6_gamma_32c['band'] = 'gamma'
pickle_in.close()



# 4 BANDS 

file = "./output/log_eeg_csp4_inter_eco_4bands_20channels"
pickle_in = open(file +".pickle", "rb")
csp4_4bands_20c = pickle.load(pickle_in)
csp4_4bands_20c['name'] = 'CSP_20c_4f'
csp4_4bands_20c['band'] = '4bands'
pickle_in.close() 

file = "./output/log_eeg_csp6_inter_eco_4bands_20channels"
pickle_in = open(file +".pickle", "rb")
csp6_4bands_20c = pickle.load(pickle_in)
csp6_4bands_20c['name'] = 'CSP_20c_6f'
csp6_4bands_20c['band'] = '4bands'
pickle_in.close()  

file = "./output/log_eeg_eco_csp_4bands"
pickle_in = open(file +".pickle", "rb")
csp4_4bands_32c = pickle.load(pickle_in)
csp4_4bands_32c['name'] = 'CSP_32c_4f'
csp4_4bands_32c['band'] = '4bands'
pickle_in.close() 

file = "./output/log_eeg_eco_csp6_4bands"
pickle_in = open(file +".pickle", "rb")
csp6_4bands_32c = pickle.load(pickle_in)
csp6_4bands_32c['name'] = 'CSP_32c_6f'
csp6_4bands_32c['band'] = '4bands'
pickle_in.close()


pickle_in.close() 

#########################################################"


df = pd.concat([no_csp_theta_20c, csp4_theta_20c, csp6_theta_20c, no_csp_theta_32c, csp4_theta_32c, csp6_theta_32c, 
                no_csp_alpha_20c, csp4_alpha_20c, csp6_alpha_20c, no_csp_alpha_32c, csp4_alpha_32c, csp6_alpha_32c,
                no_csp_beta_20c, csp4_beta_20c, csp6_beta_20c, no_csp_beta_32c, csp4_beta_32c, csp6_beta_32c, 
                no_csp_gamma_20c, csp4_gamma_20c, csp6_gamma_20c, no_csp_gamma_32c, csp4_gamma_32c, csp6_gamma_32c,
                no_csp_4bands_20c, csp4_4bands_20c, csp6_4bands_20c, no_csp_4bands_32c, csp4_4bands_32c, csp6_4bands_32c])

# val1 = df1.loc[:,['Accuracy']].values.mean().round(2)
# val2 = df2.loc[:,['Accuracy']].values.mean().round(2)

# print(len(val))
# print(val.min(), val.max(), val.mean(), np.median(val))


# df = df.rename(columns = {'Classifier':'Participants'})

acl = 55.6



f, ax = plt.subplots(figsize=(20,9))
ax = sns.boxplot(x="band", y="Accuracy", palette="coolwarm", hue="name", width=0.6, data=df, fliersize=2, showfliers = True, showmeans = True)
#ax = sns.swarmplot(x="FEATURE", y="VALUE", hue="LOAD", data=df, dodge=True, alpha=0.5, zorder=1)
ax.set(xlabel='band', ylabel='EEG Accuracy (ECO)')

plt.plot([-0.5,0,0,4.5], [acl, acl, acl, acl],  linestyle='--', linewidth=1.5, color='green')
plt.text(4.4,acl+1,str(acl),fontsize=9, color='green')

# # ax = sns.despine(offset=10, trim=True)
# plt.plot([-0.5,0,13,12], [val1, val1, val1, val1], linestyle='--', linewidth=1.5, color='blue')
# plt.text(12.5,val1-1,str(val1),fontsize=9, color='blue')

# plt.plot([-0.5,0,13,12], [val2, val2, val2, val2], linestyle='--', linewidth=1.5, color='brown')
# plt.text(12.5,val2-1,str(val2),fontsize=9, color='red')


plt.ylim([33,75])
plt.show()