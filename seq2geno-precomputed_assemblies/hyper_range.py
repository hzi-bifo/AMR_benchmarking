
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier,RandomForestClassifier,ExtraTreesClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn import svm,preprocessing
from sklearn.linear_model import LogisticRegression


def hyper_range(chosen_cl):

    if chosen_cl=='svm':
        cl=SVC(random_state=1,class_weight='balanced')
        hyper_space = [
              {
                "cl__kernel": [
                  "rbf"
                ],
                "cl__gamma": [
                  1e-2,
                  1e-3,
                  1e-4,
                  1e-5
                ],
                "cl__C": [
                  0.001,
                  0.10,
                  0.1,
                  10,
                  25,
                  50,
                  100,
                  1000
                ]
              },
              {
                "cl__kernel": [
                  "sigmoid"
                ],
                "cl__gamma": [
                  1e-2,
                  1e-3,
                  1e-4,
                  1e-5
                ],
                "cl__C": [
                  0.001,
                  0.10,
                  0.1,
                  10,
                  25,
                  50,
                  100,
                  1000
                ]
              }, {
                "cl__kernel": [
                  "linear"
                ],
                "cl__C": [
                  0.001,
                  0.10,
                  0.1,
                  10,
                  25,
                  50,
                  100,
                  1000
                ]
              }
            ]

    if chosen_cl=='lr':
        cl=LogisticRegression( random_state=1,class_weight='balanced')
        hyper_space = [
          {
            "cl__C": [
              0.001,
              0.01,
              0.015,
              0.025,
              0.10,
              0.1,
              10,
              25,
              50,
              100,
              1000
            ],
            "cl__tol": [1e-06, 1e-03, 1e-04],
            "cl__dual": [False],
             "cl__fit_intercept": [True],
             "cl__intercept_scaling": [1],
             "cl__max_iter": [1000],
             "cl__penalty": ["l2"],
          }
        ]

    if chosen_cl=='lsvm':
        cl=LinearSVC(random_state=1, max_iter=1000,class_weight='balanced')
        hyper_space = [
        {
            "cl__C": [
              0.001,
              0.01,
              0.015,
              0.025,
              0.10,
              0.1,
              10,
              25,
              50,
              100,
              1000
            ],
            "cl__tol": [1e-06,1e-03],
            "cl__dual": [True],
             "cl__intercept_scaling": [1],
             "cl__loss": ["squared_hinge"],
             "cl__multi_class": ["ovr"],
             "cl__penalty": ["l2"],
          },
          {
            "cl__C": [
              0.001,
              0.01,
              0.015,
              0.025,
              0.10,
              0.1,
              10,
              25,
              50,
              100,
              1000
            ],
            "cl__tol": [1e-06,1e-03],
            "cl__dual": [False],
             "cl__fit_intercept": [True],
             "cl__intercept_scaling": [1],
             "cl__loss": ["squared_hinge"],
             "cl__multi_class": ["ovr"],
             "cl__penalty": ["l1"],
          }

        ]


    if chosen_cl=='rf':
        cl=RandomForestClassifier(random_state=1)
        hyper_space = [
          {
            "cl__n_estimators": [100, 200, 500, 1000],
            "cl__criterion": ["entropy", "gini"],
            "cl__max_features": ["auto"],
            "cl__min_samples_split": [2,5,10],
            "cl__min_samples_leaf": [1],
            "cl__class_weight": ["balanced", None]
          }
        ]


    return hyper_space,cl
