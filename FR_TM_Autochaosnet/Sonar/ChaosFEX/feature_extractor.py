# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:20:53 2023

@author: Akhila Henry

Description:
------------
This module implements a chaos-based feature transformation for numerical input data. 
The transformation uses a truncated version of the Champernowne constant to generate 
chaotic path for each feature element and computes two features per input: 

1. **Trace Mean**: The mean of the generated path (neural trace).
2. **Firing Rate**: A simple measure based on the fraction of path values 
   exceeding a threshold (here, 0.5).

Workflow:
---------
- The `firing_time_bound()` function maps a value in [0,1] to a finite integer representing 
  how long to iterate along the Champernowne sequence.
- The Champernowne constant is precomputed up to 500 digits for reproducibility.
- For each feature value in the input matrix:
  - A path is generated from the Champernowne constant.
  - Its mean and firing rate are computed and stored.
- The final output is a transformed feature matrix with twice the number of features 
  (mean and firing rate per original feature).

Functions:
----------
- `firing_time_bound(x)`:
    Determines the number of iterations based on a quantized version of input x.
    
- `f(n)`:
    Returns the n-th truncated value of the Champernowne constant starting from digit `n`.

- `_compute_ttss_entropy(path, threshold)`:
    Computes the proportion of path values greater than the given threshold (firing rate).

- `_compute_measures(feat_mat, meas_mat)`:
    Core loop that computes both features for each input value.

- `transform(feat_mat)`:
    Main function to be called externally for transforming a 2D input feature matrix into 
    its chaos-based representation.

Dependencies:
-------------
- numpy
- numba
- math
- decimal

Returns:
--------
- A transformed 2D numpy array with shape `(samples, 2 × features)`, suitable for 
  classification tasks. 
"""


import numpy as np
import numba as nb
from decimal import getcontext
import math


# Set precision high enough for your needs
getcontext().prec = 5
#formula to find firing time bound
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

#chapernowne's constant
c=[]
#n=k+1
for i in np.arange(1,500):
    c.append(str(i))
new_c=str(''.join(c))



def f(n):
    fn=float('0.' + new_c[n:])
    return fn

    
#compute firing rate
def _compute_ttss_entropy(path, threshold):
    prob = np.count_nonzero(path > threshold) / len(path)
    return prob


def _compute_measures(feat_mat, meas_mat):
    
   
    for i in nb.prange(feat_mat.shape[0]):
       
        for j in nb.prange(feat_mat.shape[1]):
           
           
            ftb = firing_time_bound(feat_mat[i, j])
            
            
            
            if ftb != 0:
                path=np.zeros(ftb)   
                for k in np.arange(0,ftb):
                       path[k]=f(k)
            elif ftb==0:
                path=np.zeros(1)
                path[0]=f(0)
             
            meas_mat[i, j, 0] = np.mean(path)
            threshold=[0.5]
            meas_mat[i, j, 1] = _compute_ttss_entropy(path, threshold)
         
    return meas_mat


def transform(feat_mat):
 
    

    # Initialize a 3D matrix of zeroes based on input dimensions
    dimx, dimy = feat_mat.shape
    meas_mat = np.zeros([dimx, dimy, 2]) 
    # Estimate measures from the trajectory for each element in input matrix
    out = _compute_measures(feat_mat, meas_mat)

    out = out.transpose([0, 2, 1]).reshape([dimx, dimy * 2])
  
    return out

