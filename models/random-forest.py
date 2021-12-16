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

filter_list=[5,2,3,8]
for i in range(4,0,-1):
    X_train, X_test, y_train, y_test = load_data(filter_list[:i])
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=10000, num=20)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(1, 12, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [0.1,0.2,0.3,0.5,0.7]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [50,75,150]
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
    rf_random = RandomizedSearchCV(estimator=random_forest, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model

    rf_random.fit(X_train, y_train)

    base_model = RandomForestClassifier(n_estimators=200)
    base_model.fit(X_train, y_train)
    mse_base = mean_squared_error(base_model.predict(X_test), y_test)
    # print("baseline score :%f" % accuracy_score(y_test, base_model.predict(X_test)))
    print("*****base line*****")
    print("baseline mse :%f" % mean_squared_error(y_test, base_model.predict(X_test)))
    print("**********")


    best_random = rf_random.best_estimator_
    mse_best = mean_squared_error(best_random.predict(X_test), y_test)
    print("best mse :%f" % mse_best)
    test_accuracy_score=rf_random.best_estimator_.score(X_test, y_test)
    print("best accuracy score: %f" % (test_accuracy_score))
    print("accuracy_score on training set:%f" % (accuracy_score(y_train, rf_random.best_estimator_.predict(X_train))))

    joblib.dump(rf_random.best_estimator_, 'optimal_rf_rs_%d.pkl'%(int(test_accuracy_score*100)), compress=1)


    # # grid search
    # param_grid = {"max_depth": max_depth,
    #               "n_estimators": [int(x) for x in np.linspace(start=100, stop=2000, num=10)],
    #               "max_features": [1, 3, 10],
    #               "min_samples_split": [3,10],
    #               "min_samples_leaf": [3,7],
    #               "bootstrap": [True, False],
    #               "criterion": ["gini", "entropy"]}
    #
    # # grid_search= RandomForestClassifier(n_estimators=10)
    # grid_search = GridSearchCV(random_forest, param_grid=param_grid, verbose=10)
    # grid_search.fit(X_train, y_train)
    # grid_accuracy = evaluate_by_accuracy(grid_search.best_estimator_, X_test, y_test)
    # print("accuracy score of random forest model with optimal parameters: %f" % grid_accuracy)
    # joblib.dump(grid_search.best_estimator_, 'best_xgboost_rs.pkl', compress=1)
