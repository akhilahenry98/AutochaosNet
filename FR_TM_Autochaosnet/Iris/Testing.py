# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:20:53 2023

@author: Akhila Henry

Description:
------------
This script evaluates the classification performance and computational efficiency 
of the AutochaosNet algorithm and chaos-based features extracted 
via the `ChaosFEX` module.

The key steps include:
-----------------------
1. Loading the dataset.
2. Preprocessing:
   - Extracting features and labels.
   - Splitting the dataset into training and testing sets (80-20 split).
   - Normalizing all features to the range [0, 1].
3. For 50 iterations:
   - Extract chaos-based features from training and testing data using 
     `ChaosFEX.transform`.
   - Classify test samples using the `chaosnet` function, which uses cosine 
     similarity to compare test vectors with mean class vectors.
   - Record the computation time for each iteration.
4. Output:
   - Average and variance of elapsed computation time.
   - Final macro-averaged F1 Score for classification performance.

Modules Used:
-------------
- `ChaosFEX.feature_extractor`: Extracts chaos-based features using a predefined method.
- `Codes.chaosnet`: A lightweight classifier based on cosine similarity.
- `sklearn.model_selection`: For train-test split.
- `sklearn.metrics`: For computing macro F1 score.
- `numpy`, `time`, `numba`: General numerical operations and timing.

Output:
-------
- Prints:
  - Mean Elapsed Time across 50 runs
  - Variance of Elapsed Time
  - Macro F1 Score for test classification performance


"""


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from Codes import chaosnet
from sklearn.metrics import f1_score
import ChaosFEX.feature_extractor as CFX
import time

#import the IRIS Dataset from sklearn library
iris = datasets.load_iris()

#reading data and labels from the dataset
X = np.array(iris.data)
y = np.array(iris.target)
y = y.reshape(len(y),1)

#Splitting the dataset for training and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)

#Normalisation of data [0,1]
X_train_norm=(X_train-np.min(X_train,0))/(np.max(X_train,0)-np.min(X_train,0))
X_test_norm=(X_test-np.min(X_test,0))/(np.max(X_test,0)-np.min(X_test,0))




Elapsed_Time=[]

for i in np.arange(0,50):
    print('Iteration',i)
    begin=time.time()
    FEATURE_MATRIX_TRAIN = CFX.transform(X_train_norm)
    FEATURE_MATRIX_VAL = CFX.transform(X_test_norm)            
 

   
    mean_each_class, Y_PRED = chaosnet(FEATURE_MATRIX_TRAIN, y_train, FEATURE_MATRIX_VAL)
    end=time.time()
    elapsed_time=end-begin
    Elapsed_Time.append(elapsed_time)

ET_100=np.mean(Elapsed_Time)
print('Elapsed Time',ET_100)
print(np.var(Elapsed_Time))
f1 = f1_score(y_test, Y_PRED, average='macro')
print('TESTING F1 SCORE', f1)




