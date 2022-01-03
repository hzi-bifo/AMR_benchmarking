#!/usr/bin/python
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import LogisticRegression
import numpy as np

def classifer():
    # Just initialize the pipeline
    pipe = Pipeline(steps=[('estimator', SVC())])

    # Add a dict of estimator and estimator related parameters
    params_grid = [{
        'estimator': [SVC(random_state=0,kernel="rbf",class_weight='balanced')],
        'estimator__C': [0.01,0.5, 0.1],
        'estimator__gamma': [ 0.01,0.001, 0.0001,'scale','auto'],
        # 'estimator__kernel': ['rbf','linear', 'poly','sigmoid'],
        # 'estimator__kernel': ['rbf'],
    },# 'l2' only
        {
        'estimator': [LogisticRegression( random_state=0,max_iter=10000,class_weight='balanced')],
        'estimator__C': [0.01, 0.5, 0.1],
        # 'estimator__penalty': ['l1', 'l2'],
        },
        {
        'estimator':[LinearSVC(random_state=0, max_iter=10000,class_weight='balanced')],
        'estimator__C':  [0.01, 0.5, 0.1],
         }#The ‘l2’ penalty is the standard used in SVC. The ‘l1’ leads to coef_ vectors that are sparse.
    ]
    return pipe,params_grid
    # return pipeline,parameters
def classifer_lg():
    # Just initialize the pipeline
    pipe = Pipeline(steps=[('estimator',LinearSVC())])

    # Add a dict of estimator and estimator related parameters
    params_grid = [{
        'estimator':[LinearSVC(random_state=0, max_iter=10000,)],
        'estimator__C':  [0.01,0.1,0.5],
         }#The ‘l2’ penalty is the standard used in SVC. The ‘l1’ leads to coef_ vectors that are sparse.,
    ]
    return pipe,params_grid
def classifer_lsvm():
    # Just initialize the pipeline
    pipe = Pipeline(steps=[('estimator', LogisticRegression())])

    # Add a dict of estimator and estimator related parameters
    params_grid = [{
        'estimator':[LinearSVC(random_state=0, max_iter=10000,)],
        'estimator__C':  [0.01,0.1,0.5],
         }#The ‘l2’ penalty is the standard used in SVC. The ‘l1’ leads to coef_ vectors that are sparse.
    ]
    return pipe,params_grid


def classifer_svm():
    # Just initialize the pipeline
    pipe = Pipeline(steps=[('estimator', SVC())])

    # Add a dict of estimator and estimator related parameters
    params_grid = [{
        'estimator': [SVC(random_state=0,kernel="rbf")],
        'estimator__C': [0.01,0.1,0.5],
        'estimator__gamma': [ 0.01,0.001,'scale'],
        # 'estimator__kernel': ['rbf','linear', 'poly','sigmoid'],
        # 'estimator__kernel': ['rbf'],
    }# 'l2' only

    ]
    # logistic = LogisticRegression(max_iter=10000,tol=0.1)
    # pipeline = Pipeline(steps=[
    #     # ('clf_svm', SVC(random_state=0)),
    #     # ('Lsvm',LinearSVC(max_iter=10000,random_state=0)),
    #     ('logister', logistic)
    # ])
    # parameters ={
    #     # 'clf_svm__kernel': ['linear', 'rbf','poly','sigmoid'],
    #     # "clf_svm__C": [0.01,0.5, 1, 10, 100],
    #     # "clf_svm__gamma": ['scale', 'auto'],
    #     #
    #     # 'Lsvm__penalty': ['l1','l2'],
    #     # 'Lsvm__C':  [0.01,0.5, 1, 10, 100],
    #     # 'Lsvm__class_weight' :['balanced',None],
    #
    #     'logistic__C': np.logspace(-4, 4, 4),
    #     # 'logister__penalty':  ['l1', 'l2'],
    #     # 'logister__C':  [0.01,0.5, 1, 10, 100],
    #     # 'logister__class_weight':  ['balanced', None],
    #     # 'logister__solver':  ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    #     }


    return pipe,params_grid
