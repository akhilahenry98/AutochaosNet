# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:20:53 2023

@author: Akhila Henry
"""
import os
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from Trace_Codes import chaosnet
from sklearn.metrics import f1_score
import Trace_ChaosFEX.trace_feature_extractor as CFX
import pandas as pd
from numpy.fft import fft
from scipy.io import wavfile


#import the TIME SERIES FSDD Dataset 
source = "C:\\Users\\Akhila Henry\\Documents\\GitHub\\Neurochaos-Learning\\CFX+ML\Datasets\\TimeSeries\\recordings\\jackson"

#reading data and labels from the dataset
data_length = []
for fileno, filename in enumerate(os.listdir(source)):
    if(fileno<500):       
        sampling_frequency, data = wavfile.read(os.path.join(source,filename))
        if(len(data)>=3000):
            data_length.append(len(data))
            print(filename)
    
y = np.zeros((len(data_length), 1), dtype='int')
input_features = np.min(data_length)
X = np.zeros((len(data_length), input_features))
index = 0
for fileno, filename in enumerate(os.listdir(source)):
    if(fileno<500):       
        sampling_frequency, data = wavfile.read(os.path.join(source,filename))
        print(filename)
        if(len(data)>=3000):
            data_length.append(len(data))
            X[index, :] = np.abs(fft(data[0:input_features]))
            y[index, 0] = filename[0]
            index+=1




#Splitting the dataset for training and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)

#Normalisation of data [0,1]
X_train_norm=(X_train-np.min(X_train,0))/(np.max(X_train,0)-np.min(X_train,0))
X_test_norm=(X_test-np.min(X_test,0))/(np.max(X_test,0)-np.min(X_test,0))


#Testing
#PATH = os.getcwd()
#RESULT_PATH = PATH + '/CFX TUNING/RESULTS/' 
    
#INA = 0.04
#EPSILON_1 = 0.156
#DT = 0.3599999
#INA = np.load(RESULT_PATH+"/h_Q.npy")[0]
#print(INA)
#EPSILON_1 = np.load(RESULT_PATH+"/h_EPS.npy")[0]
#print(EPSILON_1)
#DT = np.load(RESULT_PATH+"/h_B.npy")[0]
#print(DT)
#F1SCORE = np.load(RESULT_PATH+"/h_fscore.npy")
#print('TRAINING F1 SCORE',F1SCORE)

FEATURE_MATRIX_TRAIN = CFX.new_transform(X_train_norm)
#print(FEATURE_MATRIX_TRAIN)
FEATURE_MATRIX_VAL = CFX.new_transform(X_test_norm)            
#plot=CFX.plot_signal(INA,DT,10000)
mean_each_class, Y_PRED = chaosnet(FEATURE_MATRIX_TRAIN, y_train, FEATURE_MATRIX_VAL)

f1 = f1_score(y_test, Y_PRED, average='macro')
print('TESTING F1 SCORE', f1)


#np.save(RESULT_PATH+"/F1SCORE_TEST.npy", np.array([f1]) )

