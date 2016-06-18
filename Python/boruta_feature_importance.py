# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12., 6.)
import seaborn as sns
sns.set()


from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score

from boruta import boruta_py


class ExecutionTime:
    ''' Usage:
    timer = ExecutionTime()
    print('Finished in {:0.2f} minutes.'.format(timer.duration()/60))
    '''
    def __init__(self):
        self.start_time = time.time()

    def duration(self):
        return time.time() - self.start_time



X = pd.read_csv('../input/trainItem_w_features_2.csv')
X.drop(['itemID_1', 'itemID_2', 'generationMethod'], axis=1, inplace=True)

y = X.pop('isDuplicate')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=27)


clf = RandomForestClassifier(n_jobs=10, max_depth=5, class_weight='auto')

feat_selector = boruta_py.BorutaPy(clf, n_estimators=1000, verbose=2)

# find all relevant features
timer = ExecutionTime()
feat_selector.fit(X.values, y.values)
print('Finished in {:0.2f} minutes.'.format(timer.duration()/60))

# check selected features
good_feats = X.columns[feat_selector.support_].tolist()
print(good_feats)


# check ranking of features
feat_selector.ranking_



clf = RandomForestClassifier(n_estimators=1000, n_jobs=10, max_depth=9, class_weight='auto')

clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)[:,1]
score = roc_auc_score(y_test, y_pred)
print(' All feats: {:0.5f}'.format(score))


clf.fit(X_train[good_feats], y_train)
y_pred = clf.predict_proba(X_test[good_feats])[:,1]
score = roc_auc_score(y_test, y_pred)
print('Good feats: {:0.5f}'.format(score))










