# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:20:53 2023

@author: Akhila Henry

Description:
------------
This module implements a feature transformation technique based on a 
hyperparameter-free approach using a truncated form of the Champernowne constant 
and  "Firing Time Bound" (FTB) formula. The primary goal is to 
generate features (trace mean) from input data for classification task under the Neurochaos Learning framework.

Key Components:
---------------
1. firing_time_bound(x):
    Computes the firing time bound (FTB) for a given input x in [0, 1], using 
    a rule-based function dependent on the first three decimal places.

2. f(n):
    Returns the n-th digit-shifted fractional value from a truncated Champernowne 
    constant sequence.

3. new_transform(feat_mat):
    Applies the transformation to a 2D feature matrix by generating a trace vector 
    for each input feature using FTB and the Champernowne sequence. The 
    feature is the mean of the trace vector (Trace Mean).

4. warmup():
    A function to trigger the JIT (Just-In-Time) compilation from Numba and validate 
    the correctness of the transformation process using a toy input.

Usage:
------
This script is intended to be imported as a module, but it also performs a JIT 
warm-up check when run directly.

Dependencies:
-------------
- numpy
- numba
- decimal
- math
"""


# Import calls
import numpy as np
import numba as nb
from decimal import getcontext
import math
# Set precision high enough for your needs
getcontext().prec = 5

#Firing Time Bound formula 
def firing_time_bound(x):
    factor=3
    if x < 0 or x > 1:
        raise ValueError("The number must be between 0 and 1, inclusive.")
    else:
        x=math.trunc(x*10**factor)/10**factor
        #print(x)
        if x==0:
            ftb=10
        elif x==1:
            ftb=0
        elif 0<x<1:
           x_dig = int(x*10**factor)
           #print(x_dig)
           sm=0
           num_digits= len(str(x_dig))
           if num_digits>1:
               for i in np.arange(1,num_digits):
                   sm=sm+10**i
                   ftb=(num_digits*x_dig)-sm-1
           elif num_digits==1 and x_dig>0:
               ftb=x_dig-1
        return ftb


#Truncate Champernowne's Constant
c=[]
for i in np.arange(1,500):
    c.append(str(i))
new_c=str(''.join(c))   


def f(n):
    fn=float('0.' + new_c[n:])
    return fn



#Feature Transformation 
def new_transform(feat_mat):
    
    path_matrix=np.zeros((feat_mat.shape[0],feat_mat.shape[1]),dtype=np.float64)
    for i in nb.prange(feat_mat.shape[0]):

        
        for j in nb.prange(feat_mat.shape[1]):  
            ftb = firing_time_bound(feat_mat[i, j])
            
            if ftb != 0:
                path=np.zeros(ftb)   
                for k in np.arange(0,ftb):
                       path[k]=f(k)
            elif ftb==0:
                path=np.zeros(1,dtype=np.float64)
                path[0]=f(0)
            path_matrix[i,j]=np.mean(path) #Calculate Tracemean
    
    return path_matrix



def warmup():
    """
    Warmup for initializing Numba's JIT compiler.
    Calls extract_feat with known and expected values.

    """
    # Initialize a feature_matrix
    feat_mat = np.array([[0.1, 0.2], [0.3, 0.4]])


    # Execute extract features
    out = new_transform(
        feat_mat
    )
    
    # Check if output matches expected value
    if out.shape == (2, 2) and out[0, 1] ==0.43173750157080265:
        print("> Numba JIT warmup successful for transform ...")
    else:
        print("> Numba JIT warmup failed for transform ...")


# Execute warmup upon import
warmup()
