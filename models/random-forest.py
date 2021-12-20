import joblib
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
from pprint import pprint

from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from utils.utils import load_data

random_forest = RandomForestRegressor(random_state=42)
print('Parameters currently in use:\n')
pprint(random_forest.get_params())

performance_score_by_filter=[]
max_accu=0
for stop_index in range(11):


    X_train, X_test, y_train, y_test = load_data(stop_index)

    # # feature selection
    # base_model = RandomForestClassifier(n_estimators=1000)
    # base_model.fit(X_train, y_train)
    # mse_base = mean_squared_error(base_model.predict(X_test), y_test)
    # print("*****base line*****")
    # print("base model accu %f" % base_model.score(X_test, y_test))
    # print("**********")
    # joblib.dump(base_model, './backup/random-forest_%d.pkl' % (stop_index), compress=1)
    #
    # performance_score_by_filter.append( base_model.score(X_test, y_test))


    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=20)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(1, 12, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2,4,8,16]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1,5,10,50, 75, 150]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   "criterion": ["gini", "entropy"]
                   }
    pprint(random_grid)

    random_forest = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=random_forest, param_distributions=random_grid, n_iter=100, cv=3,
                                   verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model

    rf_random.fit(X_train, y_train)

    test_accuracy_score = rf_random.best_estimator_.score(X_test, y_test)
    if test_accuracy_score>max_accu:
        joblib.dump(rf_random.best_estimator_, 'random-forest_%d.pkl' % (stop_index), compress=1)
        print("best accuracy score: %f" % (test_accuracy_score))
        max_accu=test_accuracy_score
        print("accuracy_score on training set:%f" % (test_accuracy_score))


# dump feature selection result
# np.save('backup/random-forest_performance_score.npy', performance_score_by_filter)

