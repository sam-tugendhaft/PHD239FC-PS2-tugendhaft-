#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:11:27 2022

@author: sam
"""


import pandas as pd 
import os
import numpy as np 
import warnings 
warnings.filterwarnings("ignore")

Directory = "/Users/sam/Documents/UC Berkeley/Classes/Second Year/Spring/Empirical Asset Pricing/Part 2/HW2/PHD239FC-PS2-tugendhaft-/Code/"
os.chdir(Directory)


# Part a
feds = pd.read_csv("../Data/feds200628.csv") 
fama_bliss = pd.read_csv("../Data/fama-bliss-yields.csv") 

epsilon = 1e-12

for column in fama_bliss:
    if column == 'Date':
        continue
    # t = int(column[1:])
    fama_bliss[column][fama_bliss[column]<=0] = epsilon
    fama_bliss[column] = 1200*np.log(1+fama_bliss[column]/12)
    

feds = feds[['Date','SVENY01','SVENY02','SVENY04','SVENY10']]
# one_month_yield = fama_bliss[['Date','y1']]
one_month_yield = fama_bliss

feds = feds.dropna()
feds['Month'] = pd.DatetimeIndex(feds['Date']).month
feds['last_obs_in_month'] = feds.Month != feds.Month.shift(-1)
feds = feds.loc[feds['last_obs_in_month']==True]
feds = feds.reset_index().drop(['index'],axis = 1) 
feds['Year'] = pd.DatetimeIndex(feds['Date']).year
feds = feds.drop(['last_obs_in_month','Date'], axis=1)

one_month_yield['Date'] = pd.to_datetime(one_month_yield['Date'], format='%Y%m%d')
one_month_yield['Month'] = pd.DatetimeIndex(one_month_yield['Date']).month
one_month_yield['Year'] = pd.DatetimeIndex(one_month_yield['Date']).year
one_month_yield = one_month_yield.drop(['Date'], axis=1)


df = pd.merge(left=feds, right=one_month_yield, left_on=['Year','Month'], right_on=['Year','Month'])

df = df[df['Year'].between(1985,2015, inclusive=True)]



df['ER1'] = df['SVENY01'] - df['y1'] - 11*(df['SVENY01'].shift(-1)-df['SVENY01'])

df['ER2'] = df['SVENY02'] - df['y1'] - 23*(df['SVENY02'].shift(-1)-df['SVENY02'])

df['ER4'] = df['SVENY04'] - df['y1'] - 47*(df['SVENY04'].shift(-1)-df['SVENY04'])

df['ER10'] = df['SVENY10'] - df['y1'] - 119* (df['SVENY10'].shift(-1)-df['SVENY10'])

df['simple1'] = np.exp((df['ER1']+df['y1'])/100)
df['simple2'] = np.exp((df['ER2']+df['y1'])/100)
df['simple4'] = np.exp((df['ER4']+df['y1'])/100)
df['simple10'] = np.exp((df['ER10']+df['y1'])/100)

df['Deltay1'] = df['SVENY01'].shift(-1)-df['SVENY01']
df['Deltay2'] = df['SVENY02'].shift(-1)-df['SVENY02']
df['Deltay4'] = df['SVENY04'].shift(-1)-df['SVENY04']
df['Deltay10'] = df['SVENY10'].shift(-1)-df['SVENY10']

df['Spread1'] = df['SVENY01'] - df['y1']
df['Spread2'] = df['SVENY02'] - df['y1']
df['Spread4'] = df['SVENY04'] - df['y1']
df['Spread10'] = df['SVENY10'] - df['y1']

df_sumstats = df.describe().loc[["mean", "std"]].T
df_sumstats = df_sumstats.tail(16)
for i in range(4):
    df_sumstats['mean'][i+4] = 100*np.log(df_sumstats['mean'][i+4])
    df_sumstats['std'][i+4] = np.nan


print(df_sumstats.to_latex(index=True)) 

# Part b
Year0 = 1985
Month0 = 1
Yearend = 2015
Monthend = 12

idx0 = fama_bliss.index[(fama_bliss['Year']==Year0) & (fama_bliss['Month'] == Month0)].values[0]
idxend = fama_bliss.index[(fama_bliss['Year']==Yearend) & (fama_bliss['Month'] == Monthend)].values[0]

fama_bliss = fama_bliss.loc[idx0:idxend+1,:]
fama_bliss = fama_bliss.reset_index().drop(['index'],axis = 1) 

fama_bliss['ER1'] = fama_bliss['y12'] - fama_bliss['y1'] - 11*(fama_bliss['y11'].shift(-1)-fama_bliss['y12'])

fama_bliss['ER2'] = fama_bliss['y24'] - fama_bliss['y1'] - 23*(fama_bliss['y23'].shift(-1)-fama_bliss['y24'])

fama_bliss['ER4'] = fama_bliss['y48'] - fama_bliss['y1'] - 47*(fama_bliss['y47'].shift(-1)-fama_bliss['y48'])

fama_bliss['ER10'] = fama_bliss['y120'] - fama_bliss['y1'] - 119*(fama_bliss['y119'].shift(-1)-fama_bliss['y120'])

fama_bliss['simple_1'] = np.exp((fama_bliss['ER1']+fama_bliss['y1'])/100)
fama_bliss['simple_2'] = np.exp((fama_bliss['ER2']+fama_bliss['y1'])/100)
fama_bliss['simple_4'] = np.exp((fama_bliss['ER4']+fama_bliss['y1'])/100)
fama_bliss['simple_10'] = np.exp((fama_bliss['ER10']+fama_bliss['y1'])/100)

fama_bliss['Deltay_cons_mat_1'] = fama_bliss['y12'].shift(-1)-fama_bliss['y12']
fama_bliss['Deltay_cons_mat_2'] = fama_bliss['y24'].shift(-1)-fama_bliss['y24']
fama_bliss['Deltay_cons_mat_4'] = fama_bliss['y48'].shift(-1)-fama_bliss['y48']
fama_bliss['Deltay_cons_mat_10'] = fama_bliss['y120'].shift(-1)-fama_bliss['y120']

fama_bliss['Deltay_noncons_mat_1'] = fama_bliss['y11'].shift(-1)-fama_bliss['y12']
fama_bliss['Deltay_noncons_mat_2'] = fama_bliss['y23'].shift(-1)-fama_bliss['y24']
fama_bliss['Deltay_noncons_mat_4'] = fama_bliss['y47'].shift(-1)-fama_bliss['y48']
fama_bliss['Deltay_noncons_mat_10'] = fama_bliss['y119'].shift(-1)-fama_bliss['y120']

fama_bliss['Spread1'] = fama_bliss['y12'] - fama_bliss['y1']
fama_bliss['Spread2'] = fama_bliss['y24'] - fama_bliss['y1']
fama_bliss['Spread4'] = fama_bliss['y48'] - fama_bliss['y1']
fama_bliss['Spread10'] = fama_bliss['y120'] - fama_bliss['y1']



fb_sumstats = fama_bliss.describe().loc[["mean", "std"]].T
fb_sumstats = fb_sumstats.tail(20)
for i in range(4):
    fb_sumstats['mean'][i+4] = 100*np.log(fb_sumstats['mean'][i+4])
    fb_sumstats['std'][i+4] = np.nan

print(fb_sumstats.to_latex(index=True)) 