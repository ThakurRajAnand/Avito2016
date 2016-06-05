# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12., 6.)
import seaborn as sns
sns.set()


class ExecutionTime:
    ''' Usage:
    timer = ExecutionTime()
    print('Finished in {:0.2f} minutes.'.format(timer.duration()/60))
    '''
    def __init__(self):
        self.start_time = time.time()

    def duration(self):
        return time.time() - self.start_time




info = pd.read_csv('../input/ItemInfo_train.csv')
pairs = pd.read_csv('../input/ItemPairs_train.csv')

info = info.drop(['title','description','attrsJSON'], axis=1)

pairs_train = pd.merge(pd.merge(pairs, info, how='inner', left_on='itemID_1', right_on='itemID'),
              info, how='inner', left_on='itemID_2', right_on='itemID')



info = pd.read_csv('../input/ItemInfo_test.csv')
pairs = pd.read_csv('../input/ItemPairs_test.csv')

info = info.drop(['title','description','attrsJSON'], axis=1)

pairs_test = pd.merge(pd.merge(pairs, info, how='inner', left_on='itemID_1', right_on='itemID'),
              info, how='inner', left_on='itemID_2', right_on='itemID')


pairs_train['images_array_x'].fillna('', inplace=True)
pairs_train['images_array_y'].fillna('', inplace=True)

pairs_test['images_array_x'].fillna('', inplace=True)
pairs_test['images_array_y'].fillna('', inplace=True)


pairs = pd.concat([pairs_train[['lat_x', 'lon_x']], pairs_test[['lat_x', 'lon_x']]])

pairs_unique = pairs.drop_duplicates(inplace=False)


from sklearn.cluster import KMeans

## stacking the x and y so they train together
#lat_lon = pd.DataFrame(columns=['lat','lon'])
#lat_lon['lat'] = np.concatenate([pairs['lat_x'].values, pairs['lat_y'].values])
#lat_lon['lon'] = np.concatenate([pairs['lon_x'].values, pairs['lon_y'].values])


km = KMeans(n_clusters=5)
km.fit(pairs_unique)

pairs_train['cluster_x_5'] = km.predict(pairs_train[['lat_x', 'lon_x']])
pairs_train['cluster_y_5'] = km.predict(pairs_train[['lat_y', 'lon_y']])

pairs_test['cluster_x_5'] = km.predict(pairs_test[['lat_x', 'lon_x']])
pairs_test['cluster_y_5'] = km.predict(pairs_test[['lat_y', 'lon_y']])


pairs_train.groupby('cluster_x_5')['isDuplicate'].mean().plot()
pairs_train.groupby('cluster_y_5')['isDuplicate'].mean().plot()
plt.show()

pairs_train['same_cluster_5'] = pairs_train['cluster_x_5'] == pairs_train['cluster_y_5']
pairs_train['same_cluster_5'] = pairs_train['same_cluster_5'].astype(np.int16)

pairs_test['same_cluster_5'] = pairs_test['cluster_x_5'] == pairs_test['cluster_y_5']
pairs_test['same_cluster_5'] = pairs_test['same_cluster_5'].astype(np.int16)


pairs_train[pairs_train['same_cluster_5']==1].groupby('cluster_x_5')['isDuplicate'].mean().plot()
pairs_train[pairs_train['same_cluster_5']==0].groupby('cluster_x_5')['isDuplicate'].mean().plot()
plt.show()



km = KMeans(n_clusters=15)
km.fit(pairs_unique)

pairs_train['cluster_x_15'] = km.predict(pairs_train[['lat_x', 'lon_x']])
pairs_train['cluster_y_15'] = km.predict(pairs_train[['lat_y', 'lon_y']])

pairs_test['cluster_x_15'] = km.predict(pairs_test[['lat_x', 'lon_x']])
pairs_test['cluster_y_15'] = km.predict(pairs_test[['lat_y', 'lon_y']])


pairs_train.groupby('cluster_x_15')['isDuplicate'].mean().plot()
pairs_train.groupby('cluster_y_15')['isDuplicate'].mean().plot()
plt.show()

pairs_train['same_cluster_15'] = pairs_train['cluster_x_15'] == pairs_train['cluster_y_15']
pairs_train['same_cluster_15'] = pairs_train['same_cluster_15'].astype(np.int16)

pairs_test['same_cluster_15'] = pairs_test['cluster_x_15'] == pairs_test['cluster_y_15']
pairs_test['same_cluster_15'] = pairs_test['same_cluster_15'].astype(np.int16)


pairs_train[pairs_train['same_cluster_15']==1].groupby('cluster_x_15')['isDuplicate'].mean().plot()
pairs_train[pairs_train['same_cluster_15']==0].groupby('cluster_x_15')['isDuplicate'].mean().plot()
plt.show()



km = KMeans(n_clusters=50)
km.fit(pairs_unique)

pairs_train['cluster_x_50'] = km.predict(pairs_train[['lat_x', 'lon_x']])
pairs_train['cluster_y_50'] = km.predict(pairs_train[['lat_y', 'lon_y']])

pairs_test['cluster_x_50'] = km.predict(pairs_test[['lat_x', 'lon_x']])
pairs_test['cluster_y_50'] = km.predict(pairs_test[['lat_y', 'lon_y']])


pairs_train.groupby('cluster_x_50')['isDuplicate'].mean().plot()
pairs_train.groupby('cluster_y_50')['isDuplicate'].mean().plot()
plt.show()

pairs_train['same_cluster_50'] = pairs_train['cluster_x_50'] == pairs_train['cluster_y_50']
pairs_train['same_cluster_50'] = pairs_train['same_cluster_50'].astype(np.int16)

pairs_test['same_cluster_50'] = pairs_test['cluster_x_50'] == pairs_test['cluster_y_50']
pairs_test['same_cluster_50'] = pairs_test['same_cluster_50'].astype(np.int16)


pairs_train[pairs_train['same_cluster_50']==1].groupby('cluster_x_50')['isDuplicate'].mean().plot()
pairs_train[pairs_train['same_cluster_50']==0].groupby('cluster_x_50')['isDuplicate'].mean().plot()
plt.show()


features = [
    'cluster_x_5',
    'cluster_y_5',
    'same_cluster_5',
    'cluster_x_15',
    'cluster_y_15',
    'same_cluster_15',
    'cluster_x_50',
    'cluster_y_50',
    'same_cluster_50'
]

pairs_train[features].to_csv('../input/clusters_train.csv')
pairs_test[features].to_csv('../input/clusters_test.csv')





