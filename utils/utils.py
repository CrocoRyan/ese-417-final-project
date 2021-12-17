import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler

FILTER_COLUMN = (3, 5, 8, 0, 4, 7, 6, 2, 9, 1, 10)


# filter column list by correlation from low to high
def load_data(stop_index):
    # load samples
    # pca = PCA(n_components=3)

    X_test = np.load('..\\data\\preprocessed\\X_test.npy')
    X_train = np.load('..\\data\\preprocessed\\X_train.npy')
    # for potential pca usage
    # X_test = pca.fit_transform(X_test)
    # X_train=pca.fit_transform(X_train)
    X_test = np.delete(X_test, FILTER_COLUMN[:stop_index], axis=1)
    X_train = np.delete(X_train, FILTER_COLUMN[:stop_index], axis=1)
    y_test = np.load('..\\data\\preprocessed\\y_test.npy')
    y_train = np.load('..\\data\\preprocessed\\y_train.npy')
    return X_train, X_test, y_train, y_test
