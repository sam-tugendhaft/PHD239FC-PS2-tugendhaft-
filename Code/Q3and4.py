#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:03:59 2022

@author: sam
"""
import numpy as np
from scipy.optimize import fsolve
import array_to_latex as a2l

## Question 3
x = [.05,.055,.057,.059,.06,.061]
yields = np.array(x)
N = yields.size
prices = np.power((1/(1+yields)),np.linspace(1,N,N))


r_tree = np.empty((N,N))
r_tree[:] = np.nan
# r_tree = np.nan([N,N])
r_tree[0,0] = yields[0]
sigma = .015

def rates(t,m):
    r_last = r_tree[0:t,t-1]
    r_new = np.empty(t+1)
    r_new[:] = np.nan
    r_new[0] = r_last[0] + m + sigma
    for i in range(t):
        r_new[i+1] = r_new[i] - 2*sigma
    return r_new

def price(t,m):
    r_new = rates(t,m)
    p = 1/(1+r_new) # initialize next period 1-period bond prices
    for i in range(t):
        # pt = np.nan(t-i) # initialize current 1-period bond price
        pt = np.empty(t-i)
        pt[:] = np.nan
        for j in range(t-i):
            pt[j] = (.5*(p[j] + p[j+1]))/(1+r_tree[j,t-i-1])
        p = pt
    return p


for i in range(N-1):
    msolved = fsolve(lambda m: price(i+1,m)-prices[i+1], .01)
    r_tree[0:i+2,i+1] = rates(i+1,msolved)
    
print(r_tree)
a2l.to_ltx(r_tree,frmt = '{:6.5f}',arraytype='tabular')    
## Question 4


# Part a
FV = 100
r = .055
T = 6
C = FV*r/(1-1/(1+r)**T)


Price = np.zeros([T+1,T+1])
for i in range(T):
    for j in range(T-i):
        Price[j,T-i-1] = (C+.5*(Price[j,T-i] + Price[j+1,T-i]))/(1+r_tree[j,T-i-1])

P0 = Price[0,0]
print(P0)

# Part b
def Principal(t):
    P = C/r*(1-1/(1+r)**t)
    return P


Price_prepay = np.zeros(Price.shape)
Exercise = np.zeros(Price.shape)
Price_prepay[:,T-1] = np.minimum(Price[:,T-1],Principal(1))
Exercise[:,T-1] = np.less(Principal(1),Price[:,T-1])
for i in range(T-1):
    for j in range(T-i-1):
        pu = Price_prepay[j,T-i-1]
        pd = Price_prepay[j+1,T-i-1]
        continue_val = (C+.5*(pu+pd))/(1+r_tree[j,T-i-2])
        exercise_val= Principal(2+i)
        Price_prepay[j,T-i-2] = min(continue_val,exercise_val)
        Exercise[j,T-i-2] = (exercise_val < continue_val)

P0_prepay = Price_prepay[0,0]                         
print(P0_prepay)
                   
# Part c and d                     
Principal_schedule = np.zeros(T+1)
Interest_schedule = np.zeros(T+1)
Princ_out = FV
for i in range(T):
    Interest_schedule[i+1] = Princ_out*r
    Principal_schedule[i+1] = C-Interest_schedule[i+1]
    Princ_out = Princ_out-Principal_schedule[i+1]                          
                          
Price_PO = np.zeros(Price.shape)
Price_PO[:,T] = Principal_schedule[T]

for i in range(T):
    for j in range(T-i):
        pu = Price_PO[j,T-i]
        pd = Price_PO[j+1,T-i]
        continue_val = (.5*(pu+pd))/(1+r_tree[j,T-i-1])
        Price_PO[j,T-i-1] = Principal_schedule[T-i-1]+ Exercise[j,i]*Principal(i+1) + (1-Exercise[j,i])*continue_val
        
P0_PO = Price_PO[0,0]
print(P0_PO)
P0_IO = P0_prepay - P0_PO 
print(P0_IO)   