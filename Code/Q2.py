#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:03:29 2022

@author: sam
"""
## Question 2
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.sandwich_covariance as sw 
from sklearn.decomposition import PCA

# Prepare data




feds = pd.read_csv("../Data/feds200628.csv") 

feds = feds.dropna()
feds['Month'] = pd.DatetimeIndex(feds['Date']).month
feds['last_obs_in_month'] = feds.Month != feds.Month.shift(-1)
feds = feds.loc[feds['last_obs_in_month']==True]
feds = feds.reset_index().drop(['index'],axis = 1) 
feds['Year'] = pd.DatetimeIndex(feds['Date']).year

feds = feds[feds['Year'].between(1985,2015, inclusive=True)]

yields = feds[['SVENY01','SVENY02','SVENY03','SVENY04','SVENY05','SVENY06','SVENY07','SVENY08','SVENY09','SVENY10']]

yields.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

for i in range(1,10):
    var_name = 'ER'+str(i+1)
    yields[var_name] = yields.iloc[:,i] - yields.iloc[:,0] - i*(yields.iloc[:,i-1].shift(-12)-yields.iloc[:,i])
    
yields.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

ER = yields.drop(['SVENY01','SVENY02','SVENY03','SVENY04','SVENY05','SVENY06','SVENY07','SVENY08','SVENY09','SVENY10'], axis=1)

ER['Ave_ER'] = ER.mean(axis = 1)
yields = yields[['SVENY01','SVENY02','SVENY03','SVENY04','SVENY05','SVENY06','SVENY07','SVENY08','SVENY09','SVENY10']]

# Part a

#Create a PCA model with five principal components
pca = PCA(5)
pca.fit(yields)
#Get the components from transforming the original data.
scores = pca.transform(yields)

X = sm.add_constant(scores)
results = sm.OLS(ER['Ave_ER'], X).fit() 

cp_factor = results.fittedvalues

print(results.summary())
betas = results.params   


Newey_West_SE = np.sqrt(np.diag(sm.stats.sandwich_covariance.cov_hac(results, nlags=18)))

Hansen_Hodrick_SE = np.sqrt(np.diag(sm.stats.sandwich_covariance.cov_hac(results, nlags=12, weights_func
= sw.weights_uniform)))

reg_results = pd.DataFrame(data=[np.array(betas), Newey_West_SE,Hansen_Hodrick_SE]).T

reg_results.columns = ['beta','Newey West SEs','Hansen Hodrick SEs']
reg_results.index = ['Constant','PC1','PC2','PC3','PC4','PC5']

print(reg_results.to_latex(index=True)) 



# Part b
rsquared_restricted = np.zeros(9)
rsquared_unrestricted = np.zeros(9)

X_restricted = sm.add_constant(cp_factor)

for i in range(9):
    # unrestricted regression
    results = sm.OLS(ER.iloc[:,i], X).fit() 
    rsquared_unrestricted[i] = results.rsquared
    # restricted regression
    results = sm.OLS(ER.iloc[:,i], X_restricted).fit() 
    rsquared_restricted[i] = results.rsquared
    
index_list = []
for i in range(9):
    name = "rx" + str(i+2)
    index_list.append(name)
    
# rsquared = pd.DataFrame(data=[rsquared_restricted, rsquared_unrestricted],index = index_list,columns = ["Restricted","Unrestricted"])
    
rsquared = pd.DataFrame(data=[rsquared_restricted, rsquared_unrestricted]).T
rsquared.columns = ["Restricted","Unrestricted"]
rsquared.index = index_list

print(rsquared.to_latex(index=True)) 