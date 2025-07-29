# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:20:53 2023

@author: Akhila Henry

Description:
------------
This module implements the ChaosNet classifier — a simple, interpretable classification 
algorithm based on the cosine similarity of transformed feature vectors.

The algorithm computes the mean feature vector (prototype) for each class in the training data 
and assigns a label to each test sample based on the highest cosine similarity to these 
class-wise mean vectors.

Function:
---------
chaosnet(traindata, trainlabel, testdata):
    - Computes class prototypes by averaging feature vectors of training samples.
    - Predicts labels for test samples based on cosine similarity to class prototypes.
    - Returns both the class mean matrix and predicted labels.

Inputs:
-------
- traindata : numpy.ndarray (2D)
    Feature matrix for training data. Shape: (n_samples, n_features)
- trainlabel : numpy.ndarray (2D, column vector)
    Corresponding labels for training data. Shape: (n_samples, 1)
- testdata : numpy.ndarray (2D)
    Feature matrix for test data. Shape: (m_samples, n_features)

Returns:
--------
- mean_each_class : numpy.ndarray (2D)
    Mean feature vector for each class (class prototype).
- predicted_label : numpy.ndarray (1D)
    Predicted class labels for the test samples.

Dependencies:
-------------
- numpy
- sklearn.metrics.pairwise.cosine_similarity
"""
import numpy as np
def chaosnet(traindata, trainlabel, testdata):
    '''
    

    Parameters
    ----------
    traindata : TYPE - Numpy 2D array
        DESCRIPTION - traindata
    trainlabel : TYPE - Numpy 2D array
        DESCRIPTION - Trainlabel
    testdata : TYPE - Numpy 2D array
        DESCRIPTION - testdata

    Returns
    -------
    mean_each_class : Numpy 2D array
        DESCRIPTION - mean representation vector of each class
    predicted_label : TYPE - numpy 1D array
        DESCRIPTION - predicted label

    '''
    from sklearn.metrics.pairwise import cosine_similarity
    NUM_FEATURES = traindata.shape[1]
    NUM_CLASSES = len(np.unique(trainlabel))
    mean_each_class = np.zeros((NUM_CLASSES, NUM_FEATURES))
    
    for label in range(0, NUM_CLASSES):
        
        mean_each_class[label, :] = np.mean(traindata[(trainlabel == label)[:,0], :], axis=0)
    predicted_label = np.argmax(cosine_similarity(testdata, mean_each_class), axis = 1)

    return mean_each_class, predicted_label


