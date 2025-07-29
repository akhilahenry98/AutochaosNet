# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:20:53 2023

@author: Akhila Henry

Description:
------------
This script evaluates the performance of the AutochaosNet classifier on the 
dataset using features extracted from the Trace Chaos 
Feature Extractor (`Trace_ChaosFEX`).

The workflow involves:
-----------------------
1. Loading the dataset .
2. Preprocessing the data by:
   - Extracting features and labels.
3. Splitting the dataset into training and testing sets (80-20 split).
4. Normalizing all features to the range [0, 1].
5. Applying the `new_transform()` function from `Trace_ChaosFEX` to extract 
   chaos-based features.
6. Performing classification using the `chaosnet` function, which calculates 
   cosine similarity between test vectors and class mean prototypes.
7. Evaluating the classifier using the **macro-averaged F1 score**.

Dependencies:
-------------
- numpy
- pandas
- sklearn 
- Trace_Codes.chaosnet (custom classifier)
- Trace_ChaosFEX.trace_feature_extractor (custom feature transformation)

Output:
-------
- Prints the macro F1 score on the test set, which reflects classification performance 
  across both classes.


"""

import numpy as np
from sklearn.model_selection import train_test_split
from Trace_Codes import chaosnet
from sklearn.metrics import f1_score
import Trace_ChaosFEX.trace_feature_extractor as CFX
import pandas as pd



#import the IONOSPHERE Dataset 
ionosphere = np.array(pd.read_csv('ionosphere_data.txt', sep=",", header=None))


#reading data and labels from the dataset
X, y = ionosphere[:,range(0,ionosphere.shape[1]-1)], ionosphere[:,ionosphere.shape[1]-1].astype(str)


#Norm: B -> 0;  G -> 1
y = y.reshape(len(y),1)
y = np.char.replace(y, 'b', '0', count=None)
y = np.char.replace(y, 'g', '1', count=None)
y = y.astype(int)



#Splitting the dataset for training and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)
X_train_norm = X_train
X_train_norm[:,range(2,X_train.shape[1])] = (X_train[:,range(2,X_train.shape[1])]-np.min(X_train[:,range(2,X_train.shape[1])],0))/(np.max(X_train[:,range(2,X_train.shape[1])],0)-np.min(X_train[:,range(2,X_train.shape[1])],0))
X_train_norm = X_train_norm.astype(float)
X_test_norm = X_test
X_test_norm[:,range(2,X_test.shape[1])] = (X_test[:,range(2,X_test.shape[1])]-np.min(X_test[:,range(2,X_test.shape[1])],0))/(np.max(X_test[:,range(2,X_test.shape[1])],0)-np.min(X_test[:,range(2,X_test.shape[1])],0))
X_test_norm = X_test_norm.astype(float)



FEATURE_MATRIX_TRAIN = CFX.new_transform(X_train_norm)

FEATURE_MATRIX_VAL = CFX.new_transform(X_test_norm)            

mean_each_class, Y_PRED = chaosnet(FEATURE_MATRIX_TRAIN, y_train, FEATURE_MATRIX_VAL)

f1 = f1_score(y_test, Y_PRED, average='macro')
print('TESTING F1 SCORE', f1)




