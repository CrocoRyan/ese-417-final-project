import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import os

from sklearn.preprocessing import StandardScaler


def load_data(filter_column=(5,2,3,8,0)):
    # load samples
    pca = PCA(n_components=3)

    X_test = np.load('..\\data\\preprocessed\\X_test.npy')
    X_train = np.load('..\\data\\preprocessed\\X_train.npy')
    # for potential pca usage
    # X_test = pca.fit_transform(X_test)
    # X_train=pca.fit_transform(X_train)
    X_test=np.delete(X_test,filter_column,axis=1)
    X_train=np.delete(X_train,filter_column,axis=1)
    y_test = np.load('..\\data\\preprocessed\\y_test.npy')
    y_train = np.load('..\\data\\preprocessed\\y_train.npy')
    return X_train, X_test, y_train, y_test
