import numpy as np

import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import pandas as pd
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


def plot_performance_by_filter():
    results = [i for i in os.listdir('../models') if i.endswith('score.npy')]
    base = pd.DataFrame(columns=['performance', 'filter_type'])
    tmp = ['density', 'chlorides', 'volatile acidity', 'total sulfur dioxide', 'fixed acidity', 'pH',
           'residual sugar', 'sulphates', 'citric acid', 'free sulfur dioxide','no filter']
    tmp.reverse()
    base['filter_type'] = tmp
    f, ax = plt.subplots(figsize=(11, 9))
    for result_file in results:
        performance_data = np.load(os.path.join('../models', result_file))
        model_name = result_file[:result_file.find('_')].upper()
        base[model_name] = performance_data
        sns.lineplot(x='filter_type', y=model_name, data=base, legend='brief', label=model_name).set(
            title='performance by filter selections')
    plt.xlabel('filter_type (accumulative) ---→')
    plt.xticks(rotation=45)
    plt.ylabel('accuracy')

    f.savefig('performance_by_filtering.jpg', dpi=100, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    plot_performance_by_filter()
