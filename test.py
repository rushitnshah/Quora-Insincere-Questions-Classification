#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 17:44:10 2018

@author: rushit
"""
import os
import re
import gc
import seaborn as sns

#Linear Algebra
import numpy as np # linear algebra
from numpy.random import random_sample
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
from collections import Counter

#For ML models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score, precision_score, recall_score
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#To deal with imbalanced data
from imblearn.over_sampling import SMOTENC
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RandomUnderSampler

#Deep learning library for RNN
from keras.layers import Input
from keras import Model
from keras.preprocessing import sequence,text
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D,Flatten,MaxPooling1D,GRU,SpatialDropout1D,Bidirectional
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import Callback

#For preprocessing text data
import nltk
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer,WordNetLemmatizer
stemmer=SnowballStemmer('english')
lemma=WordNetLemmatizer()
from string import punctuation

#To save variables/models
import pickle


class BaselineModel:
    def __init__(self,type='majority'):
        self.type = type
        print("Instance of {0} classifier".format(self.type))
    def fit(self, X , y):
        if self.type=='majority':
            counts = Counter(y)
            freq_max = 0
            for label, freq in counts.items():
                if freq>freq_max:
                    freq_max = freq
                    self.label = label
        elif self.type=='random':
                self.label = 0
                
    def predict(self, X):
        predictions = np.zeros((X.shape[0],1))
        if self.type == 'majority':
            predictions = np.ones((X.shape[0],1)) * self.label
        elif self.type == 'random':
            predictions = random_sample((X.shape[0],1))
            predictions[predictions>=0.5]=1
            predictions[predictions<0.5]=0
        
        return predictions

def line_search_f1_score(y_score, y_test):
    max_f1_score = 0
    opt_threshold = 0
    for threshold in [i*0.01 for i in range(100)]:
        y_preds = y_score > threshold
        score = f1_score(y_preds, y_test)
        if max_f1_score < score:
            max_f1_score = score
            opt_threshold = threshold
    return max_f1_score, opt_threshold

def line_search_acc_score(y_score, y_test):
    max_acc_score = 0
    opt_threshold = 0
    for threshold in [i*0.01 for i in range(100)]:
        y_preds = y_score > threshold
        score = accuracy_score(y_preds, y_test)
        if max_acc_score < score:
            max_acc_score = score
            opt_threshold = threshold
    return max_acc_score, opt_threshold

class Metrics(Callback):
    def __init__(self):
        self.best_threshold = 0.5
        self.best_f1_score = 0
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.best_f1_score = 0
    def on_epoch_end(self, epoch, logs={}):
         idx = np.random.randint(0,self.validation_data[0].shape[0],1000)
         val_predict = (np.asarray(self.model.predict(self.validation_data[0][idx], verbose=1))).round()
         val_targ = self.validation_data[1][idx]
         #_val_f1 = f1_score(val_targ, val_predict)
         _val_f1, threshold = line_search_f1_score(val_targ, val_predict)
         if _val_f1 > self.best_f1_score:
                self.best_f1_score = _val_f1
         self.best_threshold = threshold
         _val_recall = recall_score(val_targ, val_predict)
         _val_precision = precision_score(val_targ, val_predict)
         self.val_f1s.append(_val_f1)
         self.val_recalls.append(_val_recall)
         self.val_precisions.append(_val_precision)
         print(" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
         return
 
metric = Metrics()

print("\n\nLoading Validation/Test Data....")
filename = 'X_test_bal'
infile = open(filename,'rb')
X_test_bal2 = pickle.load(infile)
infile.close()

filename = 'X_val_bal'
infile = open(filename,'rb')
X_val_bal2 = pickle.load(infile)
infile.close()

filename = 'y_val_bal'
infile = open(filename,'rb')
y_val_bal2 = pickle.load(infile)
infile.close()

filename = 'y_test_bal'
infile = open(filename,'rb')
y_test_bal2 = pickle.load(infile)
infile.close()

print("\n\nLoading Trained Models....")
filename = 'modelMaj_bal'
infile = open(filename,'rb')
modelMaj_bal2 = pickle.load(infile)
infile.close()

filename = 'modelRand_bal'
infile = open(filename,'rb')
modelRand_bal2 = pickle.load(infile)
infile.close()

filename = 'modelLR_bal'
infile = open(filename,'rb')
modelLR_bal2 = pickle.load(infile)
infile.close()

filename = 'modelRF_bal'
infile = open(filename,'rb')
modelRF_bal2 = pickle.load(infile)
infile.close()

filename = 'modelLSTM_bal'
infile = open(filename,'rb')
modelLSTM_bal2 = pickle.load(infile)
infile.close()

print("\n\nPredicting validation/test data labels using trained models....")
predictions_val  = modelRand_bal2.predict(X_val_bal2)
predictions_test = modelRand_bal2.predict(X_test_bal2)
f1_val_rand    , threshold_val  = line_search_f1_score(predictions_val , y_val_bal2)
acc_val_rand   , threshold_val  = line_search_acc_score(predictions_val, y_val_bal2)
f1_test_rand   , threshold_test  = line_search_f1_score(predictions_test , y_test_bal2)
acc_test_rand  , threshold_test  = line_search_acc_score(predictions_test, y_test_bal2)


predictions_val  = modelMaj_bal2.predict(X_val_bal2)
predictions_test = modelMaj_bal2.predict(X_test_bal2)
f1_val_Maj    , threshold_val  = line_search_f1_score(predictions_val , y_val_bal2)
acc_val_Maj   , threshold_val  = line_search_acc_score(predictions_val, y_val_bal2)
f1_test_Maj   , threshold_test  = line_search_f1_score(predictions_test , y_test_bal2)
acc_test_Maj  , threshold_test  = line_search_acc_score(predictions_test, y_test_bal2)

predictions_val  = modelLR_bal2.predict(X_val_bal2)
predictions_test = modelLR_bal2.predict(X_test_bal2)
f1_val_LR    , threshold_val  = line_search_f1_score(predictions_val , y_val_bal2)
acc_val_LR   , threshold_val  = line_search_acc_score(predictions_val, y_val_bal2)
f1_test_LR   , threshold_test  = line_search_f1_score(predictions_test , y_test_bal2)
acc_test_LR  , threshold_test  = line_search_acc_score(predictions_test, y_test_bal2)

predictions_val  = modelRF_bal2.predict(X_val_bal2)
predictions_test = modelRF_bal2.predict(X_test_bal2)
f1_val_RF    , threshold_val  = line_search_f1_score(predictions_val , y_val_bal2)
acc_val_RF   , threshold_val  = line_search_acc_score(predictions_val, y_val_bal2)
f1_test_RF   , threshold_test  = line_search_f1_score(predictions_test , y_test_bal2)
acc_test_RF  , threshold_test  = line_search_acc_score(predictions_test, y_test_bal2)

predictions_val  = modelLSTM_bal2.predict(X_val_bal2)
predictions_test = modelLSTM_bal2.predict(X_test_bal2)
f1_val_LSTM    , threshold_val  = line_search_f1_score(predictions_val , y_val_bal2)
acc_val_LSTM   , threshold_val  = line_search_acc_score(predictions_val, y_val_bal2)
f1_test_LSTM   , threshold_test  = line_search_f1_score(predictions_test , y_test_bal2)
acc_test_LSTM  , threshold_test  = line_search_acc_score(predictions_test, y_test_bal2)


print("Tabulated Summary of Results")
print("\n\n")
print("Results for Validation Data (balanced)")
print(" ________________________________________________")
print("|   Classifier            |   Acc     |    F1    |")
print("|_________________________|___________|__________|")
print("|   Majority Classifier   |   {:.4f}  |   {:.4f} |".format(acc_val_Maj,f1_val_Maj))
print("|   Random Classifier     |   {:.4f}  |   {:.4f} |".format(acc_val_rand,f1_val_rand))
print("|   Logistic Regression   |   {:.4f}  |   {:.4f} |".format(acc_val_LR,f1_val_LR))
print("|   Random Forest         |   {:.4f}  |   {:.4f} |".format(acc_val_RF,f1_val_RF))
print("|   LSTM                  |   {:.4f}  |   {:.4f} |".format(acc_val_LSTM,f1_val_LSTM)) 
print("|_________________________|___________|__________|")

print("\n\n")
print("Results for Test Data (balanced)")
print(" ________________________________________________")
print("|   Classifier            |   Acc     |    F1    |")
print("|_________________________|___________|__________|")
print("|   Majority Classifier   |   {:.4f}  |   {:.4f} |".format(acc_test_Maj,f1_test_Maj))
print("|   Random Classifier     |   {:.4f}  |   {:.4f} |".format(acc_test_rand,f1_test_rand))
print("|   Logistic Regression   |   {:.4f}  |   {:.4f} |".format(acc_test_LR,f1_test_LR))
print("|   Random Forest         |   {:.4f}  |   {:.4f} |".format(acc_test_RF,f1_test_RF))
print("|   LSTM                  |   {:.4f}  |   {:.4f} |".format(acc_test_LSTM,f1_test_LSTM)) 
print("|_________________________|___________|__________|")


print("Bar Chart Summary of Results")
# data to plot
n_groups = 4
maj  = (acc_val_Maj, acc_test_Maj,f1_val_Maj, f1_test_Maj)
rand = (acc_val_rand, acc_test_rand,f1_val_rand, f1_test_rand)
lr   = (acc_val_LR, acc_test_LR ,f1_val_LR, f1_test_LR)
rf   = (acc_val_RF, acc_test_RF,f1_val_RF, f1_test_RF)
lstm = (acc_val_LSTM, acc_test_LSTM, f1_val_LSTM, f1_test_LSTM)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.1
opacity = 0.8
 
rects1 = plt.bar(index, maj, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Majority Classifier (Baseline)') 
rects2 = plt.bar(index + bar_width, rand, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Random Classiifier (Baseline)')
rects3 = plt.bar(index + (2*bar_width), lr, bar_width,
                 alpha=opacity,
                 color='y',
                 label='Logistic Regression')
rects4 = plt.bar(index + (3*bar_width), rf, bar_width,
                 alpha=opacity,
                 color='k',
                 label='Random Forest')
rects5 = plt.bar(index + (4*bar_width),lstm, bar_width,
                 alpha=opacity,
                 color='r',
                 label='LSTM')
 
plt.xlabel('Data Set/Metric')
plt.ylabel('Accuracy')
plt.title('Results Summary (Balanced Data)')
plt.xticks(index + 2*bar_width, ('Validation Set\n(Accuracy)', 'Validation Set\n(F1_Score)','Test Set\n(Accuracy)','Test Set\n(F1_Score)'))
plt.legend(loc="lower right")
plt.show()
fig.savefig("Result_Summary.png",dpi=600)