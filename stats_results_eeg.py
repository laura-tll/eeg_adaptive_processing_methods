#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:46:06 2020

@author: c.ponzoni
"""


import pandas as pd
import pickle
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


eeg ="./output/log_eeg_eco_csp6_beta_20channels.pickle"
pickle_in = open(eeg, "rb")
lh_eeg = pickle.load(pickle_in)
pickle_in.close()

lh_eeg = lh_eeg.reset_index(drop=True)

lh_eeg['Participants'] = lh_eeg['Participants'].astype(int)
mean_all = []
for i in range (13):
    mean_all.append(lh_eeg.loc[lh_eeg['Participants']==i,["Accuracy"]].values.mean())
    
    
eeg2 ="../../coding/phd/ml/output/log_eeg_eco_LinearDiscriminantAnalysis.pickle"
pickle_in = open(eeg2, "rb")
lh_eeg2 = pickle.load(pickle_in)
pickle_in.close()

lh_eeg2 = lh_eeg2.reset_index(drop=True)

lh_eeg2['Classifier'] = lh_eeg2['Classifier'].astype(int)
mean_all2 = []
for i in range (13):
    mean_all2.append(lh_eeg2.loc[lh_eeg2['Classifier']==i,["Accuracy"]].values.mean())
    
    
# test if normal distribution (if p-value > 0.05 = true)
print("shapiro test for ", "csp and no csp")
res_csp = stats.shapiro(mean_all)
print(res_csp)
res_nocsp = stats.shapiro(mean_all2)
print(res_nocsp)


# if normally distributed : T-test for significant difference
print("T-test HR")
res = stats.ttest_rel(mean_all, mean_all2)
print(res)

# if not : Wilcoxon test for significant difference
print("Wilcoxon test")
res = stats.wilcoxon(mean_all, mean_all2)
print(res)


lh_eeg['name'] = 'CSP_beta_20c'
lh_eeg2['name'] = 'No_CSP_betagammaEI'

lh_eeg2.rename(columns={"Classifier": "Participants"}, inplace=True)


# df = pd.concat([lh_eeg, lh_eeg2])

# f, ax = plt.subplots(figsize=(15,9))
# ax = sns.boxplot(x="Participants", y="Accuracy", palette="coolwarm", hue="name", data=df, width=0.5, fliersize=2, showfliers = False)
# #ax = sns.swarmplot(x="FEATURE", y="VALUE", hue="LOAD", data=df, dodge=True, alpha=0.5, zorder=1)
# ax.set(xlabel='Participants', ylabel='EEG Accuracy')
# # ax = sns.despine(offset=10, trim=True)
# # plt.plot([0,0,13,13], [med, med, med, med], linewidth=1, color='blue')
# # plt.text(11.2,med+1,"Median:" +str(med),fontsize=9, color='blue')
# plt.ylim([35,70])
# plt.show()

df = pd.DataFrame(mean_all,columns=['Accuracy'])
name = ['csp_beta_20c','csp_beta_20c','csp_beta_20c','csp_beta_20c','csp_beta_20c','csp_beta_20c',
        'csp_beta_20c','csp_beta_20c','csp_beta_20c','csp_beta_20c','csp_beta_20c','csp_beta_20c','csp_beta_20c']
df['name'] = name

df['Accuracy'].mean()

df['Accuracy'].median()

df2 = pd.DataFrame(mean_all2,columns=['Accuracy'])
name = ['no_csp_betagammaEI','no_csp_betagammaEI','no_csp_betagammaEI','no_csp_betagammaEI','no_csp_betagammaEI',
        'no_csp_betagammaEI','no_csp_betagammaEI','no_csp_betagammaEI','no_csp_betagammaEI','no_csp_betagammaEI',
        'no_csp_betagammaEI','no_csp_betagammaEI','no_csp_betagammaEI']
df2['name'] = name

df2['Accuracy'].mean()
df2['Accuracy'].median()

dff = pd.concat([df, df2])

sns.set(style="whitegrid", palette="pastel", color_codes=True)
sns.boxenplot(x="name", y="Accuracy", data=dff, showmeans=True)
#sns.despine(left=True)
#plt.legend(loc='upper left')
# plt.plot([0 , 0, 1, 1], [102, 105, 105, 102], linewidth=1, color='k')
# plt.text(0.5,107,'***',fontsize=12)
plt.show()
# plt.savefig('hr_load.png')
# plt.clf()

sns.boxenplot(x="name", y="Accuracy", data=df2)
#sns.despine(left=True)
plt.plot([0 , 0, 1, 1], [78, 81, 81, 78], linewidth=1, color='k')
plt.text(0.5,84,'*',fontsize=12)
#plt.legend(loc='upper left')
plt.show()
# plt.savefig('hrv_load.png')
plt.clf()