# author: Karlos Muradyan
# date: 2020-01-24

'''This script does ML analysis by training multiple models, doing hyperparameter
tuning and reporting the results in a csv file.

Usage: 
    ml_analysis.py --train_csv=<train_csv> --test_csv=<test_csv> --output_csv=<output_csv> --output_png=<output_png>
    ml_analysis.py --test_csv=<test_csv> --output_csv=<output_csv>
    ml_analysis.py --train_csv=<train_csv> --output_csv=<output_csv>
    ml_analysis.py --train_csv=<train_csv> --test_csv=<test_csv>
    ml_analysis.py --train_csv=<train_csv>
    ml_analysis.py --test_csv=<test_csv>
    ml_analysis.py --output_csv=<output_csv>
    ml_analysis.py

Options:
--train_csv=<train_csv>         csv path for training [Default: ./data/clean/bank_train.csv].
--test_csv=<test_csv>           csv path for testing [Default: ./data/clean/bank_test.csv].
--output_csv=<output_csv>       csv path for outputting the result of training and hyperparameter tuning.
                                [Default: ./reports/training_report.csv]
--output_png=<output_png>       png path for outputting the result of figure containing all trainings
                                [Default: ./reports/training_report.png]
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import lightgbm as lgb
from docopt import docopt
from itertools import accumulate
import os
import altair as alt
import selenium

os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
np.random.RandomState(42)
np.random.seed(2020)

def check_filepath(save_dir):
    """
    Checks if all subfolders of save_dir exist or not. If not, creates

    Parameters
    ----------
    save_dir: str containing the path where the files hould be saved

    Returns:
    None

    Usage
    -----
    check_filepath('./unknown_dir')
    """
    for subdir in accumulate(save_dir.split('/'), lambda x, y: os.path.join(x, y)):
        if not os.path.exists(subdir):
            os.mkdir(subdir)
            print(f"Directory {subdir} Created ")


def train_lgb(X_train, y_train, X_test, y_test, epochs = 1000, early_stopping = 100):
    """
    Function that handles the training process of Lightgbm.

    Parameters
    ----------
    X_train: 2D numpy.ndarray containing predictors for training
    y_train: 1D numpy.ndarray containing response for training
    X_test: 2D numpy.ndarray containing predictors for testing
    y_test: 1D numpy.ndarray containing response for testing
    epochs: positive integer specifying number of weak learners in the model
            Default: 1000
    early_stopping: represents number of epochs that is required to pass without
                model improvement to stop the training earlier. Default: 100

    Returns
    -------
    tuple: (model, parameters, Train F1 score, Test F1 score, Test accuracy)
        
    Examples
    --------
    >>>train_lgb(X_train, y_train, X_test, y_test) 
    (<lightgbm.basic.Booster at 0x7fd9f06b3470>,
     {'learning_rate': 0.01, 'lambda_l2': 0.5},
     0.6242424242424243,
     0.3972602739726027,
     0.9027624309392265)
    """
    # Defining custom f1 score for LightGBM model
    def lgb_f1_score(y_hat, data):
        y_true = data.get_label()
        y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
        return 'f1', f1_score(y_true, y_hat), True

    # Defining validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)

    # Creating dataset objects for LightGBM
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid)

    # For all possible parameters see https://lightgbm.readthedocs.io/en/latest/Parameters.html
    params = {
        "learning_rate" : 0.1,
        "lambda_l1": 0.5,
        "max_depth": 64,
        "num_leaves": 32,
        "bagging_fraction" : 0.9,
        "bagging_freq": 3,
        "bagging_seed": 42,
        "seed": 42
    }

    # Training model
    model = lgb.train(params,
                       lgb_train,
                       valid_sets=[lgb_train, lgb_valid],
                       feval = lgb_f1_score,
                       num_boost_round=epochs,
                       early_stopping_rounds=early_stopping,
                       verbose_eval=False)

    # Getting train F1 score
    lgb_train_preds = np.round(model.predict(X_train))
    train_f1 = f1_score(y_train, lgb_train_preds)

    # Getting test F1 score
    lgb_test_preds = np.round(model.predict(X_test))
    test_f1 = f1_score(y_test, lgb_test_preds)
    
    # Getting test accuracy
    test_acc = sum(lgb_test_preds == y_test)/len(y_test)

    # Returning whatever is important
    return model, params, train_f1, test_f1, test_acc, [train_f1], [test_f1]


def hyperparameter_tuning_and_report(classifier, parameters, X, y, X_test=None, y_test=None, scoring='f1'):
    """
    Tunes hyperparameters of given model from the list of parameters.
    
    Uses GridSearchCV to find best hyperparameter. Optionally can
    calculate F1 score and accuracy of test set. Scoring function
    can be changed that GridSearchCV is using. Default scoring function
    is F1.

    Parameters
    ----------
    X: 2D numpy.ndarray containing predictors for training
    y: 1D numpy.ndarray containing response for training
    X_test: 2D numpy.ndarray containing predictors for testing. If None, test
            scores will not be computed. Default: None
    y_test: 1D numpy.ndarray containing response for testing. If None, test
            scores will not be computed. Default: None
    scoring: Scoring function used in GridSearchCV. Default is 'f1'. For all
            possibilities, please check documentation of GridSearchCV.

    Returns
    -------
    tuple: (best model, best parameters, Train F1 score, Test F1 score, Test accuracy, Mean Train scores, Mean Test scores)
        
    Examples
    --------
    >>>hyperparameter_tuning_and_report(LogisticRegression(),
                                        {'penalty': ['l1', 'l2'], 'C': [0.1, 1, 10]},
                                        X_train, y_train, X_test, y_test)
    (LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                    warm_start=False),
     {'C': 10, 'penalty': 'l2'},
     0.43209194041441296,
     0.3776223776223776,
     0.901657458563536,
     array([       nan, 0.37217253,        nan, 0.44838868,        nan,
            0.45292401]),
     array([       nan, 0.34413357,        nan, 0.42585911,        nan,
            0.43209194]))
    """
    # Find the best model
    try:
        grid_search = GridSearchCV(classifier, parameters, n_jobs=-1, scoring=scoring, return_train_score=True)
        grid_search.fit(X, y)
    except ValueError:
        pass
    
    test_f1 = None
    test_accuracy = None

    y_train_pred = grid_search.predict(X)
    train_report = classification_report(y, y_train_pred, output_dict=True)

    # Test best model on test set and produce classification report
    if X_test is not None and y_test is not None:
        y_test_pred = grid_search.predict(X_test)
        report = classification_report(y_test, y_test_pred, output_dict=True)
        test_f1 =    report['1.0']['f1-score'], 
        test_accuracy =    report['accuracy'], 

    # Return whatever is important
    return (grid_search.best_estimator_, 
            grid_search.best_params_, 
            train_report['1.0']['f1-score'], 
            test_f1,
            test_accuracy,
            grid_search.cv_results_['mean_train_score'], 
            grid_search.cv_results_['mean_test_score'])

def generate_csv_and_figure_reports(arr, csv_filepath, figure_filepath):
    """
    Generates csv report from the results obtained from hyperparameter tuning.

    Given the array of results of hyperparameter_tuning_and_report for different
    models, generates csv file and saves in the given path.

    Parameters
    ----------
    arr: 1D array containing a tuple returned from hyperparameter_tuning_and_report or
        train_lgb() function.
    csv_filepath: str containing path where the report should be saved.
    figure_filepath: str containing path where the figure report should be saved. Should have extension .png

    Returns
    -------
    None

    Notes:
    Please be sure that arr is array of tuples.
    """
    names = []
    best_params = []
    train_f1s = []
    test_f1s = []
    test_accuracies = []

    train_score_results = []
    test_score_results = []
    score_res_names = []

    # Gathering names of the models and other information
    for model in tqdm(arr):
        names.append(model[0].__class__.__name__)
        best_params.append(model[1])
        train_f1s.append(model[2])
        test_f1s.append(model[3])
        test_accuracies.append(model[4])
    
        if model[0].__class__.__name__ != 'Booster':
            train_score_results.extend(model[5])
            test_score_results.extend(model[6])
            score_res_names.extend([model[0].__class__.__name__]*len(model[5]))

    # Creating dataframe from the results
    csv_report = pd.DataFrame({'Model name': names,
                               'Best parameters': best_params,
                              'Train F1': train_f1s,
                              'Test F1': test_f1s,
                              'Test accuracies': test_accuracies})
    
    # Creating dataframe from the results for figure generation
    figure_report = pd.DataFrame({'models': score_res_names,
                                 'train_scores': train_score_results,
                                 'test_scores': test_score_results})

    # Check for existance of a filepath
    check_filepath(csv_filepath.rsplit('/', 1)[0])
    check_filepath(figure_filepath.rsplit('/', 1)[0])

    # Saving the report
    csv_report.to_csv(csv_filepath)

    # Saving figure
    alt.Chart(figure_report).mark_circle(size=100).encode(
        x = alt.X('test_scores', axis = alt.Axis(title='Test F1 score')),
        y = alt.Y('train_scores',  axis = alt.Axis(title='Test F1 score')),
        color = 'models').properties(
        title = 'Train and Test scores of all methods tested').\
    configure_axis(
        labelFontSize=15,
        titleFontSize=15
    ).\
    configure_legend(labelFontSize = 15,
                     titleFontSize=15).\
    save(figure_filepath)


def read_data_and_split(train_csv_path = '../data/clean/bank_train.csv',
                        test_csv_path = '../data/clean/bank_test.csv'):
    """
    Reads the data from the given paths and returns predictors and response
    variables separately for train and test sets

    Parameters
    ----------
    train_csv_path: str containing the path of train csv. Default: '../data/clean/bank_train.csv'
    test_csv_path: str containing the path of test csv. Default: '../data/clean/bank_test.csv'

    Returns
    -------
    tuple: (X_train, y_train, X_test, y_test)
    """
    try:
        train_ds = pd.read_csv(train_csv_path)
        test_ds = pd.read_csv(test_csv_path)
    except (FileNotFoundError) as e:
        print('Please check train and test filepaths')
        raise(e)

    try:
        X_train, y_train = train_ds.drop('y_yes', axis=1), train_ds['y_yes']
        X_test, y_test = test_ds.drop('y_yes', axis=1), test_ds['y_yes']
    except KeyError:
        print('Corrupted csv files. Please check the columns')
        raise KeyError

    return X_train, y_train, X_test, y_test

def main(train_csv, test_csv, output_csv, output_png):
    try:
        X_train, y_train, X_test, y_test = read_data_and_split(train_csv, test_csv)
    except:
        return

    # Define models that should be passed to hyperparameter tuning
    models = [LogisticRegression(class_weight = 'balanced', random_state=42),
             SVC(class_weight = 'balanced', random_state=42),
             RandomForestClassifier(class_weight = 'balanced', random_state=42)
             ]
             

    # Define parameters that should be tested for each of the models. 
    # Note: Be sure that indices of models and its parameters correspond.
    # Note2: Each model should have some valid dictionalry associated with it.
    parameters = [
            [{'solver': ['saga', 'liblinear'], 'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10]},
             {'solver': ['lbfgs', 'newton-cg', 'sag'], 'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10]}],
            [{'C': [0.01, 0.1, 1, 10], 'kernel': ['rbf']},
             {'C': [0.01, 0.1, 1, 10], 'kernel': ['poly'], 'degree': [2, 3, 4]}],
            {'n_estimators': [25, 50, 75], 'max_depth': [None, 16, 32], 'criterion': ['gini', 'entropy']}
    ]

    if len(models) != len(parameters):
        print('Check models and corresponding parameters. Each model should have a dictionary of parameters to test.')
        return

    # Performing hyperparameter tuning and getting reports of each of the models
    # defined above
    all_results = []
    for model_id, model in enumerate(tqdm(models)):
        model_params = parameters[model_id]
        res = hyperparameter_tuning_and_report(model, model_params, X_train, y_train, X_test, y_test)
        all_results.append(res)

    # Training LightGBM model and appending results to other results
    lgb_model_res = train_lgb(X_train, y_train, X_test, y_test)
    all_results.append(lgb_model_res)

    # Generating and saving model results
    generate_csv_and_figure_reports(all_results, output_csv, output_png)

if __name__ == '__main__':
    opt = docopt(__doc__)
    main(opt["--train_csv"], opt["--test_csv"], opt["--output_csv"], opt["--output_png"])
