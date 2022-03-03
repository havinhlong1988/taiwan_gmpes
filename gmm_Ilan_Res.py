# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 01:41:31 2021

@author: ben
"""

# change working directory
import os 
os.chdir('G:/I_Lan_GMPE_2020/Python_ILAN')


import pandas as pd
import numpy as np
import pylib.pylib_P21ILAN as P21ILAN

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 120
import matplotlib.gridspec as gridspec
#warning
import warnings
warnings.filterwarnings("ignore")




# Select ground motion data 
df = pd.read_csv('G:/MyStudy_2019/Ground Motion Database/GMC 2018 Database_v9_Final/SSHAC_GM_v9_20180620_VH.csv')

ds = df[df['STA_ID'].str.contains('ILA0')][df['MW']>=4][df['Hyp.Depth']<=35][df['Rrup']<=200]
# select only necessary columns

col_name1 = ['EQSN','MW','Ztor','Dip','Rake','STA_Lon_X','STA_Lat_Y','fault.type','eq.type',
            'Vs30','Z1_0','Rrup','Lowest_Usable_Freq_H1','Lowest_Usable_Freq_H2']

col_name2 = [a for a in ds.columns if ("S" in a) & ("T" in a)]

col_name = col_name1 + col_name2

ds = ds[col_name]

# Remove negative values if it has
## select psa for all period
# psa_raw = ds[[a for a in ds.columns if ("S" in a) & ("T" in a)]].drop(columns="STA_ID")
# reomve vertical psa 
for col in ds.columns:
    if "_V" in col:
        del ds[col]
psa_clean = ds[[a for a in ds.columns if ("." in a) & ("T" in a)]]
# reomove negative numer in dataframe        
# psa_clean = psa_raw[(psa_raw > 0).all(1)].iloc[:,3:]

# select the associated period
periods = np.array([float(s.translate({ord(i): None for i in 'TS'})) for s in psa_clean.columns])
periods_col = psa_clean.columns


# calculate the residual
tns = ['T0.010S','T0.200S','T1.000S','T3.000S']

ip = periods_col.str.contains('T0.010S')

print(periods[ip])


# Initiate the gmm model
gmmp21 = P21ILAN.Phung2021h_ILAN()

ypred = []
for i in range(ds.shape[0]):
    y = gmmp21.Ph2021_ILAN_p(periods[ip], ds.MW.iloc[i], ds.Ztor.iloc[i], ds.Rake.iloc[i],
                     ds.Dip.iloc[i], ds.Rrup.iloc[i], ds.Vs30.iloc[i], -999)[0]
    ypred.append(y)
ypred = np.stack(ypred).flatten()

# Compute the total residual
res = np.log(ds[periods_col[ip]].values.flatten()) - np.log(ypred)


fig, ax = plt.subplots(figsize = (10, 6))
ax.semilogx(ds.Rrup,res,'o')


# Perform mixed effect regression to split up the total residual into event term and within event residual
# this is the random intercept model
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import seaborn as sns; sns.set()

# Create a dataframe to store the variables 
dres = ds[["EQSN","MW","Rrup","Vs30","Z1_0","fault.type","eq.type"]]
dres["res"] = res 

md = smf.mixedlm("res ~ 1", dres, groups = dres["EQSN"])
mdf = md.fit()

dBe = pd.DataFrame(mdf.random_effects).T
dBe.columns = ["dBe"]
dres = dres.merge(dBe, left_on = "EQSN", right_index = True)
dres["Wes"] = mdf.resid
dres["ck"] = np.ones(dres.shape[0])*mdf.fe_params[0]


fig, ax = plt.subplots(2,1, figsize = (16, 20))
ax[0].scatter(dres.MW,dres.dBe,s=300,c='b',marker='s')
ax[0].set_xlabel("Mw",fontsize = 20)
ax[0].set_ylabel("dBe", fontsize = 20)
ax[0].set(ylim=(-2,2))

ax[1].scatter(dres.Rrup, dres.Wes, s= 200, c='b', marker="o")
ax[1].set(xscale="log")
ax[1].set_xlabel("Rrup",fontsize = 20)
ax[1].set_ylabel("Wes", fontsize = 20)
ax[1].set(ylim=(-3,3))



# =============================================================================
# f,ax =  plt.subplots(figsize=(10,6))
# ax.set(xscale="log")
# g = sns.regplot(x = "Rrup", y = "res", data = dres)
# g.set(ylim=(-3, 3))
# =============================================================================


# =============================================================================
# g = sns.lmplot(x = "Rrup", y = "res", hue = "fault.type", col="fault.type", data = dres,
#                col_wrap=2, height=3)
# g.set(xscale="log")
# g.set(ylim=(-3, 3))
# =============================================================================
