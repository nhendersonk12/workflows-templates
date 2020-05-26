import time
import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from logger import update_predict_log, update_train_log

# model specific variables (iterate the version and note with each change)
if not os.path.exists(os.path.join(".", "models")):
    os.mkdir("models")

MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "SVM on toy data"
SAVED_MODEL = os.path.join("models", "model-{}.joblib".format(
    re.sub("\.", "_", str(MODEL_VERSION))))


def fetch_data():
    """example function to fetch data for training"""

    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target

    return(X, y)


def model_train(test=False):
    """function to train model

    The 'test' flag when set to 'True':
        (1) subsets the data and serializes a test version
        (2) specifies that the use of the 'test' log file

    The diabetes data set is already small so the subset is shown as an example
    """

    # start timer for runtime
    time_start = time.time()

    # data ingestion
    X, y = fetch_data()

    # subset the data to enable faster unittests
    if test:
        n_samples = int(np.round(0.9 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(
            X.shape[0]), n_samples, replace=False).astype(int)
        y = y[subset_indices]
        X = X[subset_indices, :]

    # Perform a train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    # Specify parameters and model
    param_grid = {
        'kernel': ['linear', 'rbf'],
        'C': [0.001, 0.01, 0.1, 1.0, 10.0],
        'gamma': [0.1, 0.01, 0.001]
    }

    print("... grid searching")
    clf = svm.SVC(probability=True)
    grid = GridSearchCV(clf, param_grid=param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    params = grid.best_params_

    # fit model on training data
    clf = svm.SVC(**params, probability=True)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    eval_test = classification_report(y_test, y_pred, output_dict=True)

    # retrain using all data
    clf.fit(X, y)

    if test:
        print("... saving test version of model")
        joblib.dump(clf, os.path.join("models", "test.joblib"))
    else:
        print(f"... saving model: {SAVED_MODEL}")
        joblib.dump(clf, SAVED_MODEL)

        print("... saving latest data")
        data_file = os.path.join("models", 'latest-train.npz')
        if isinstance(X, pd.DataFrame):
            args = {'y': y, 'X': X.to_numpy()}
        else:
            args = {'y': y, 'X': X}
        np.savez_compressed(data_file, **args)

    m, s = divmod(time.time() - time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    # update the log file
    update_train_log(X.shape, eval_test, runtime,
                     MODEL_VERSION, MODEL_VERSION_NOTE, test=test)


def model_predict(query, model=None, test=False):
    """example function to predict from model"""

    # start timer for runtime
    time_start = time.time()

    # input checks
    if isinstance(query, list):
        query = np.array([query])

    # load model if needed
    if not model:
        model = model_load()

    # output checking
    if len(query.shape) == 1:
        query = query.reshape(1, -1)

    # make prediction and gather data for log entry
    y_pred = model.predict(query)
    y_proba = None
    if 'predict_proba' in dir(model) and model.probability == True:
        y_proba = model.predict_proba(query)

    m, s = divmod(time.time() - time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    # update the log file
    for i in range(query.shape[0]):
        update_predict_log(y_pred[i], y_proba[i], runtime, query.shape,
                           MODEL_VERSION, test=test)

    return({'y_pred': y_pred, 'y_proba': y_proba})


def model_load():
    """example function to load model"""

    if not os.path.exists(SAVED_MODEL):
        exc = f"Model '{SAVED_MODEL}' cannot be found did you train the model?"
        raise Exception(exc)

    model = joblib.load(SAVED_MODEL)
    return(model)


if __name__ == "__main__":
    # basic test procedure for model.py

    # train the model
    model_train(test=True)

    # load the model
    model = model_load()

    # example predict
    for query in [[6.1, 2.8], [7.7, 2.5], [5.8, 3.8]]:
        result = model_predict(query, model, test=True)
        y_pred = result['y_pred']
        print("predicted: {}".format(y_pred))
