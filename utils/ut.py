#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split
import os

def evaluate_by_accuracy(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy

def load_data():
    # load samples
    X_test = np.load('..\\data\\preprocessed\\X_test.npy')
    X_train = np.load('..\\data\\preprocessed\\X_train.npy')
    y_test = np.load('..\\data\\preprocessed\\y_test.npy')
    y_train = np.load('..\\data\\preprocessed\\y_train.npy')
    return X_train, X_test, y_train, y_test


# In[ ]:




