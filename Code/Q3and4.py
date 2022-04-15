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

# Create yields vector
x = [.05,.055,.057,.059,.06,.061]
yields = np.array(x)
N = yields.size

# Calculate the bond prices
prices = np.power((1/(1+yields)),np.linspace(1,N,N))

# Initialize our tree
r_tree = np.empty((N,N))
r_tree[:] = np.nan

# Initial value in the tree is the spot short rate
r_tree[0,0] = yields[0]

# Ho-Lee parameter given
sigma = .015

# This function calculates the short rates at time t given the rates at time t-1, with ho-lee drift m. Initialize highest value of r as previous highest value + m + sigma. Every other value in the tree will sequentially decrease by 2sigma

def rates(t,m):
    r_last = r_tree[0:t,t-1]
    r_new = np.empty(t+1)
    r_new[:] = np.nan
    r_new[0] = r_last[0] + m + sigma
    for i in range(t):
        r_new[i+1] = r_new[i] - 2*sigma
    return r_new

# This value calculates the price of a t-year bond, given all of the interest rates up to t-1, and specifying the last drift value of m.

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

# Iteratively calibrate the values of m (and therefore the interest rate values in the tree) by matching bond prices. Fill in the tree iteratively with the calibrated m.
    
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

# Calculate mortgage payment
C = FV*r/(1-1/(1+r)**T)

# Initialize tree for prices of the mortgage. Note that prices are post-mortgage-payment
Price = np.zeros([T+1,T+1])

# Using the interest rate tree, calculate the value of the mortage by discounting back expected next period value plus discounted mortgage payment
for i in range(T):
    for j in range(T-i):
        Price[j,T-i-1] = (C+.5*(Price[j,T-i] + Price[j+1,T-i]))/(1+r_tree[j,T-i-1])

# Grab value of the mortgage
P0 = Price[0,0]
print(P0)

# Part b

# This function calculates the amount of principal outstanding with t years to maturity
def Principal(t):
    P = C/r*(1-1/(1+r)**t)
    return P

# Initialize tree of value of prepayable mortgage
Price_prepay = np.zeros(Price.shape)

# Initialize tree of 1s and 0s, 1s at times when you decide to exercise the prepayment option.
Exercise = np.zeros(Price.shape)

# Boundary condition for prepayable mortgage. The value is the minimum between the nonprepaybale mortgage and the amount of principal outstanding at time T-1.
Price_prepay[:,T-1] = np.minimum(Price[:,T-1],Principal(1))

# Values for exercising at time T-1
Exercise[:,T-1] = np.less(Principal(1),Price[:,T-1])

# Fill in tree
for i in range(T-1):
    for j in range(T-i-1):
        # continuation value in up state
        pu = Price_prepay[j,T-i-1] 
        # continuation value in down state
        pd = Price_prepay[j+1,T-i-1] 
        # Discounted continuation value plus discounted mortage payment
        continue_val = (C+.5*(pu+pd))/(1+r_tree[j,T-i-2]) 
        # Exercise value is just the amount of principal outstanding
        exercise_val= Principal(2+i)
        # Homeowner prepays if that is less value than not prepaying
        Price_prepay[j,T-i-2] = min(continue_val,exercise_val)
        Exercise[j,T-i-2] = (exercise_val < continue_val)

# Grab value of prepayable mortgage
P0_prepay = Price_prepay[0,0]                         
print(P0_prepay)
                   
# Part c and d

# First grab principal and interest payment schedules                     
Principal_schedule = np.zeros(T+1)
Interest_schedule = np.zeros(T+1)
Princ_out = FV
for i in range(T):
    Interest_schedule[i+1] = Princ_out*r
    Principal_schedule[i+1] = C-Interest_schedule[i+1]
    Princ_out = Princ_out-Principal_schedule[i+1]                          

# Initialize value of Principal Only (PO) tree. Note that these values are pre-mortgage payment
Price_PO = np.zeros(Price.shape)
# If you make it to time T without prepaying the value is the value of the last scheduled principal payment
Price_PO[:,T] = Principal_schedule[T]

for i in range(T):
    for j in range(T-i):
        # continuation value in the up state
        pu = Price_PO[j,T-i]
        # continuation value in the down state
        pd = Price_PO[j+1,T-i]
        # discounted continuation value
        continue_val = (.5*(pu+pd))/(1+r_tree[j,T-i-1])
        # price today is equal to the scheduled principal payment plus principal outstanding if mortgage is prepayed, otherwise it's plus the continuation value if they don't prepay
        Price_PO[j,T-i-1] = Principal_schedule[T-i-1]+ Exercise[j,i]*Principal(i+1) + (1-Exercise[j,i])*continue_val

# Grab value of Principal Only Security    
P0_PO = Price_PO[0,0]
print(P0_PO)

# Grab value of Interest Only Security, which is difference between value of prepayable mortgage and Principal Only Security
P0_IO = P0_prepay - P0_PO 
print(P0_IO)   