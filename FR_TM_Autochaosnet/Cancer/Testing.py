# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:20:53 2023

@author: Akhila Henry

Description:
------------
This script performs classification on the dataset using the ChaosNet algorithm 
with chaos-based features extracted via the ChaosFEX transform.

Steps involved:
---------------
1. Loads the Haberman dataset.
2. Extracts features (X) and labels (y)
3. Reshapes labels to a 2D array for compatibility.
4. Splits the dataset into training and testing sets using an 80-20 ratio.
5. Applies min-max normalization to scale features to the [0, 1] range.
6. Transforms both training and testing feature matrices using `ChaosFEX.transform`.
7. Applies the `chaosnet` function to classify the transformed test data using cosine similarity 
   with class-wise mean vectors.
8. Evaluates performance using macro-averaged F1 Score.

Modules Used:
-------------
- `pandas`: For reading the CSV file.
- `numpy`: For numerical operations.
- `sklearn.model_selection`: For splitting the dataset.
- `sklearn.metrics`: For computing F1 score.
- `ChaosFEX.feature_extractor`: For chaos-based feature extraction.
- `Codes.chaosnet`: Classifier based on cosine similarity.

Output:
-------
- Prints the macro F1 Score on the test set.


"""

import numpy as np
from sklearn.model_selection import train_test_split
from Codes import chaosnet
from sklearn.metrics import f1_score
import ChaosFEX.feature_extractor as CFX
import pandas as pd


#import the BREAST CANCER WISCONSIN Dataset 
breastcancer = np.array(pd.read_csv('breast-cancer-wisconsin.txt', sep=",", header=None))


#reading data and labels from the dataset
X, y = breastcancer[:,range(2,breastcancer.shape[1])], breastcancer[:,1].astype(str)
y = np.char.replace(y, 'M', '0', count=None)
y = np.char.replace(y, 'B', '1', count=None)
y = y.astype(int)
y = y.reshape(len(y),1)
X = X.astype(float)

#Splitting the dataset for training and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)

#Normalisation of data [0,1]
X_train_norm=(X_train-np.min(X_train,0))/(np.max(X_train,0)-np.min(X_train,0))
X_test_norm=(X_test-np.min(X_test,0))/(np.max(X_test,0)-np.min(X_test,0))




FEATURE_MATRIX_TRAIN = CFX.transform(X_train_norm)
FEATURE_MATRIX_VAL = CFX.transform(X_test_norm)            
#plot=CFX.plot_signal(INA,DT,10000)     
mean_each_class, Y_PRED = chaosnet(FEATURE_MATRIX_TRAIN, y_train, FEATURE_MATRIX_VAL)

f1 = f1_score(y_test, Y_PRED, average='macro')
print('TESTING F1 SCORE', f1)




