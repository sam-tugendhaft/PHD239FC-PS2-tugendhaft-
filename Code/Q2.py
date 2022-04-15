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
import os

# Make this whatever code directory you're in, with data stored in ../Data/
Directory = "/Users/sam/Documents/UC Berkeley/Classes/Second Year/Spring/Empirical Asset Pricing/Part 2/HW2/PHD239FC-PS2-tugendhaft-/Code/"
os.chdir(Directory)

# Prepare data



# Load data
feds = pd.read_csv("../Data/feds200628.csv") 

# Create year and month variables, keep end of month observations
feds = feds.dropna()
feds['Month'] = pd.DatetimeIndex(feds['Date']).month
feds['last_obs_in_month'] = feds.Month != feds.Month.shift(-1)
feds = feds.loc[feds['last_obs_in_month']==True]
feds = feds.reset_index().drop(['index'],axis = 1) 
feds['Year'] = pd.DatetimeIndex(feds['Date']).year

# Keep data between 1985-2015
feds = feds[feds['Year'].between(1985,2015, inclusive=True)]

# Grab relevant yields
yields = feds[['SVENY01','SVENY02','SVENY03','SVENY04','SVENY05','SVENY06','SVENY07','SVENY08','SVENY09','SVENY10']]

# Drop NaN rows
yields.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

# Create Excess Returns using change in yield over 12 months
for i in range(1,10):
    var_name = 'ER'+str(i+1)
    yields[var_name] = yields.iloc[:,i] - yields.iloc[:,0] - i*(yields.iloc[:,i-1].shift(-12)-yields.iloc[:,i])
    
yields.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

# Create dataframe of excess returns 
ER = yields.drop(['SVENY01','SVENY02','SVENY03','SVENY04','SVENY05','SVENY06','SVENY07','SVENY08','SVENY09','SVENY10'], axis=1)

ER['Ave_ER'] = ER.mean(axis = 1)

# Create dataframe of yields
yields = yields[['SVENY01','SVENY02','SVENY03','SVENY04','SVENY05','SVENY06','SVENY07','SVENY08','SVENY09','SVENY10']]

# Part a

#Create a PCA model with five principal components
pca = PCA(5)
# Fit the model to the 1-10 year yields
pca.fit(yields)
#Get the components from transforming the original data.
scores = pca.transform(yields)

# Regress average excess returns on the components
X = sm.add_constant(scores)
results = sm.OLS(ER['Ave_ER'], X).fit() 

# store the fitted value for part b
cp_factor = results.fittedvalues

print(results.summary())
betas = results.params   

# Calculate standard errors
Newey_West_SE = np.sqrt(np.diag(sm.stats.sandwich_covariance.cov_hac(results, nlags=18)))

Hansen_Hodrick_SE = np.sqrt(np.diag(sm.stats.sandwich_covariance.cov_hac(results, nlags=12, weights_func
= sw.weights_uniform)))

# Store results for table
reg_results = pd.DataFrame(data=[np.array(betas), Newey_West_SE,Hansen_Hodrick_SE]).T

reg_results.columns = ['beta','Newey West SEs','Hansen Hodrick SEs']
reg_results.index = ['Constant','PC1','PC2','PC3','PC4','PC5']

print(reg_results.to_latex(index=True)) 



# Part b

# Initialize R-squared values
rsquared_restricted = np.zeros(9)
rsquared_unrestricted = np.zeros(9)

X_restricted = sm.add_constant(cp_factor)

# Loop over excess returns and regress on all 5 factors in unrestricted regression and on the fitted value of average excess return in the restricted regression
for i in range(9):
    # unrestricted regression
    results = sm.OLS(ER.iloc[:,i], X).fit() 
    rsquared_unrestricted[i] = results.rsquared
    # restricted regression
    results = sm.OLS(ER.iloc[:,i], X_restricted).fit() 
    rsquared_restricted[i] = results.rsquared

# Store results in dataframe for latex
index_list = []
for i in range(9):
    name = "rx" + str(i+2)
    index_list.append(name)
    
rsquared = pd.DataFrame(data=[rsquared_restricted, rsquared_unrestricted]).T
rsquared.columns = ["Restricted","Unrestricted"]
rsquared.index = index_list

print(rsquared.to_latex(index=True)) 