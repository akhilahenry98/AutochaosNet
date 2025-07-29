# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:25:00 2023

@author: Akhila Henry

Description:
------------
This script demonstrates the classification performance of the AutochaosNet algorithm 
on the dataset using features extracted from the Trace Chaos Feature Extractor (Trace_ChaosFEX). 
It evaluates classification accuracy via the macro F1-score and measures computation time 
over multiple iterations.

Workflow:
---------
1. Load the dataset.
2. Normalize the features to the [0, 1] range.
3. Split the data into training and testing sets (80-20 split).
4. Apply Trace_ChaosFEX's `new_transform()` method to extract chaos-based features.
5. Classify the test data using the ChaosNet algorithm (based on cosine similarity).
6. Repeat the classification for 50 iterations and record the elapsed time for each run.
7. Compute and report:
   - Mean computational time
   - Variance of computational time
   - Macro F1-score

Dependencies:
-------------
- numpy
- sklearn 
- Trace_Codes.chaosnet (custom classification function)
- Trace_ChaosFEX.trace_feature_extractor (custom feature transformation)
- time

Output:
-------
- Average elapsed time for transformation and classification
- Variance of elapsed time
- Macro-averaged F1 Score


"""
import numpy as np
from sklearn.model_selection import train_test_split
from Trace_Codes import chaosnet
from sklearn.metrics import f1_score
import Trace_ChaosFEX.trace_feature_extractor as CFX
import pandas as pd
import time


#import the SEEDS Dataset 
seeds = np.array(pd.read_csv('seeds_dataset.txt', sep="\t" ,header=None))


#reading data and labels from the dataset
X, y = seeds[:,range(0,seeds.shape[1]-1)], seeds[:,seeds.shape[1]-1]
y = y.reshape(len(y),1).astype(str)
y = np.char.replace(y, '1.0', '0', count=None)
y = np.char.replace(y, '2.0', '1', count=None)
y = np.char.replace(y, '3.0', '2', count=None)
y = y.astype(int)

#Splitting the dataset for training and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)

#Normalisation of data [0,1]
X_train_norm=(X_train-np.min(X_train,0))/(np.max(X_train,0)-np.min(X_train,0))
X_test_norm=(X_test-np.min(X_test,0))/(np.max(X_test,0)-np.min(X_test,0))



Elapsed_Time=[]

for i in np.arange(0,50):
   print('Iteration',i)
   begin=time.time() 
   FEATURE_MATRIX_TRAIN = CFX.new_transform(X_train_norm)
   FEATURE_MATRIX_VAL = CFX.new_transform(X_test_norm)            
   mean_each_class, Y_PRED = chaosnet(FEATURE_MATRIX_TRAIN, y_train, FEATURE_MATRIX_VAL)

   f1 = f1_score(y_test, Y_PRED, average='macro')
   end=time.time()
   elapsed_time=end-begin
   Elapsed_Time.append(elapsed_time)

ET_100=np.mean(Elapsed_Time)
print('Elapsed Time',ET_100)
print(np.var(Elapsed_Time))
f1 = f1_score(y_test, Y_PRED, average='macro')
print('TESTING F1 SCORE', f1)



