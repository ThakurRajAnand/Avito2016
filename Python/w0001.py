# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12., 6.)
import seaborn as sns
sns.set()


train = pd.read_csv('../input/ItemInfo_train.csv', index_col='itemID')
test = pd.read_csv('../input/ItemInfo_test.csv', index_col='itemID')

train_idx = train.index.tolist()
test_idx = test.index.tolist()

train_test = pd.concat([train, test])

trainItem = pd.read_csv('../input/ItemPairs_train.csv')
testItem =  pd.read_csv('../input/ItemPairs_test.csv', index_col='id')

imageHashes = pd.read_csv('../input/image_features_dhash.csv', index_col='image')
imageHashes_flip = pd.read_csv('../input/image_features_dhash_flip.csv', index_col='image')





def convert_image_strings_to_lists(itemID):
    images = train_test.loc[train_test['itemID']==itemID, 'images_array'].values[0]
    if not str(images)=='nan':
        images = images.split(',')
        images = [int(i.strip()) for i in images]
        return images
    else:
        return []

train_test['image_list'] = np.nan
train_test = train_test.reset_index()
train_test['image_list'] = train_test['itemID'].apply(lambda x: convert_image_strings_to_lists(x))
train_test = train_test.set_index('itemID')
train_test.to_csv('../input/train_test.csv')




def get_dhash_count(itemID_1, itemID_2):
    dhash_count = 0
    images_1 = train_test.loc[itemID_1, 'image_list']
    images_2 = train_test.loc[itemID_2, 'image_list']
    if str(images_1) == 'nan' or str(images_2) == 'nan':
        return np.nan
    else:
        if images_1 and images_2:
            for i1 in images_1:
                for i2 in images_2:
                    if imageHashes.loc[i1].values == imageHashes.loc[i2].values:
                        dhash_count += 1
            return dhash_count
        else:
            return np.nan

trainItem['dhash_count'] = np.nan
testItem['dhash_count'] = np.nan

trainItem['dhash_count'] = trainItem.apply(lambda x: get_dhash_count(int(x[0]), int(x[1])), axis=1)
testItem['dhash_count'] = testItem.apply(lambda x: get_dhash_count(x[0], x[1]), axis=1)

trainItem['dhash_count'].fillna(-1, inplace=True)
testItem['dhash_count'].fillna(-1, inplace=True)

trainItem['dhash_count'] = trainItem['dhash_count'].astype(np.int16)
testItem['dhash_count'] = testItem['dhash_count'].astype(np.int16)

trainItem.to_csv('../input/trainItem_w_features.csv')
testItem.to_csv('../input/testItem_w_features.csv')

# Very strong dependence on dhash_count
trainItem.groupby('dhash_count')['isDuplicate'].mean().plot()
plt.title('isDuplicate vs dhash_count', fontsize='16')
plt.show()


def get_dhash_flip_count(itemID_1, itemID_2):
    dhash_count = 0
    images_1 = train_test.loc[itemID_1, 'image_list']
    images_2 = train_test.loc[itemID_2, 'image_list']
    if str(images_1) == 'nan' or str(images_2) == 'nan':
        return np.nan
    else:
        if images_1 and images_2:
            for i1 in images_1:
                for i2 in images_2:
                    if imageHashes.loc[i1].values == imageHashes_flip.loc[i2].values:
                        dhash_count += 1
            return dhash_count
        else:
            return np.nan

trainItem['dhash_flip_count'] = np.nan
testItem['dhash_flip_count'] = np.nan

trainItem['dhash_flip_count'] = trainItem.apply(lambda x: get_dhash_flip_count(x[0], x[1]), axis=1)
testItem['dhash_flip_count'] = testItem.apply(lambda x: get_dhash_flip_count(x[0], x[1]), axis=1)

trainItem['dhash_flip_count'].fillna(-1, inplace=True)
testItem['dhash_flip_count'].fillna(-1, inplace=True)

trainItem['dhash_flip_count'] = trainItem['dhash_flip_count'].astype(np.int16)
testItem['dhash_flip_count'] = testItem['dhash_flip_count'].astype(np.int16)

trainItem.to_csv('../input/trainItem_w_features.csv')
testItem.to_csv('../input/testItem_w_features.csv')

trainItem.groupby('dhash_flip_count')['isDuplicate'].mean().plot()
plt.title('isDuplicate vs dhash_flip_count', fontsize='16')
plt.show()




def get_image_count(itemID):
    images = train_test.loc[itemID, 'image_list']
    if not str(images)=='nan':
        return len(images)
    else:
        return np.nan

trainItem['image1_count'] = np.nan
testItem['image1_count'] = np.nan

trainItem['image1_count'] = trainItem.apply(lambda x: get_image_count(x[0]), axis=1)
testItem['image1_count'] = testItem.apply(lambda x: get_image_count(x[0]), axis=1)

trainItem['image2_count'] = np.nan
testItem['image2_count'] = np.nan

trainItem['image2_count'] = trainItem.apply(lambda x: get_image_count(x[1]), axis=1)
testItem['image2_count'] = testItem.apply(lambda x: get_image_count(x[1]), axis=1)

trainItem['image1_count'].fillna(-1, inplace=True)
testItem['image1_count'].fillna(-1, inplace=True)

trainItem['image2_count'].fillna(-1, inplace=True)
testItem['image2_count'].fillna(-1, inplace=True)

trainItem.to_csv('../input/trainItem_w_features.csv')
testItem.to_csv('../input/testItem_w_features.csv')

trainItem.groupby('image1_count')['isDuplicate'].mean().plot()
plt.title('isDuplicate vs image1_count', fontsize='16')
plt.show()

trainItem.groupby('image2_count')['isDuplicate'].mean().plot()
plt.title('isDuplicate vs image2_count', fontsize='16')
plt.show()


trainItem['img_count_diff'] = np.abs(trainItem['image2_count'] - trainItem['image1_count'])
testItem['img_count_diff'] = np.abs(testItem['image2_count'] - testItem['image1_count'])

trainItem.loc[trainItem['img_count_diff']>5, 'img_count_diff'] = 5
testItem.loc[testItem['img_count_diff']>5, 'img_count_diff'] = 5

trainItem.groupby('img_count_diff')['isDuplicate'].mean().plot()
plt.title('isDuplicate vs img_count_diff', fontsize='16')
plt.show()

trainItem.to_csv('../input/trainItem_w_features.csv')
testItem.to_csv('../input/testItem_w_features.csv')



imageHashes['1'] = 1
imageHashes.groupby('dhash').sort('1')
gb = imageHashes.groupby('dhash')
gb_count = gb.count()
imageHashes.drop('1', axis=1, inplace=True)

white = u'0000000000000000000000000000000000000000000000000000000000000000'
top_200 = gb_count.nlargest(201, '1').index.tolist()
top_200.remove(white)






def white_count(itemID):
    white_count = 0
    images = train_test.loc[itemID, 'image_list']
    if not str(images)=='nan':
        for i in images:
            if imageHashes.loc[i].values == white:
                white_count += 1
        return white_count
    else:
        -1

trainItem['white_count1'] = np.nan
testItem['white_count1'] = np.nan

trainItem['white_count1'] = trainItem.apply(lambda x: white_count(x[0]), axis=1)
testItem['white_count1'] = testItem.apply(lambda x: white_count(x[0]), axis=1)

trainItem['white_count2'] = np.nan
testItem['white_count2'] = np.nan

trainItem['white_count2'] = trainItem.apply(lambda x: white_count(x[1]), axis=1)
testItem['white_count2'] = testItem.apply(lambda x: white_count(x[1]), axis=1)

trainItem['white_count1'].fillna(-1, inplace=True)
testItem['white_count1'].fillna(-1, inplace=True)

trainItem['white_count2'].fillna(-1, inplace=True)
testItem['white_count2'].fillna(-1, inplace=True)

trainItem.to_csv('../input/trainItem_w_features.csv')
testItem.to_csv('../input/testItem_w_features.csv')

trainItem.groupby('white_count1')['isDuplicate'].mean().plot()
plt.title('isDuplicate vs white_count1', fontsize='16')
plt.show()

trainItem.groupby('white_count2')['isDuplicate'].mean().plot()
plt.title('isDuplicate vs white_count2', fontsize='16')
plt.show()

trainItem['white_count_diff'] = np.abs(trainItem['white_count1'] - trainItem['white_count2'])
testItem['white_count_diff'] = np.abs(testItem['white_count1'] - testItem['white_count2'])

#trainItem.loc[trainItem['img_count_diff']>5, 'img_count_diff'] = 5
#testItem.loc[testItem['img_count_diff']>5, 'img_count_diff'] = 5

trainItem.groupby('white_count_diff')['isDuplicate'].mean().plot()
plt.title('isDuplicate vs white_count_diff', fontsize='16')
plt.show()

trainItem.to_csv('../input/trainItem_w_features.csv')
testItem.to_csv('../input/testItem_w_features.csv')



top_200 = frozenset(top_200)

def top_200_count(itemID):
    top_200_count = 0
    images = train_test.loc[itemID, 'image_list']
    if not str(images)=='nan' and images:
        for i in images:
            if imageHashes.loc[i].values[0] in top_200:
                top_200_count += 1
        return top_200_count
    else:
        -1

trainItem['top_200_count1'] = np.nan
testItem['top_200_count1'] = np.nan

trainItem['top_200_count1'] = trainItem.apply(lambda x: top_200_count(x[0]), axis=1)
testItem['top_200_count1'] = testItem.apply(lambda x: top_200_count(x[0]), axis=1)

trainItem['top_200_count2'] = np.nan
testItem['top_200_count2'] = np.nan

trainItem['top_200_count2'] = trainItem.apply(lambda x: top_200_count(x[1]), axis=1)
testItem['top_200_count2'] = testItem.apply(lambda x: top_200_count(x[1]), axis=1)

trainItem['top_200_count1'].fillna(-1, inplace=True)
testItem['top_200_count1'].fillna(-1, inplace=True)

trainItem['top_200_count2'].fillna(-1, inplace=True)
testItem['top_200_count2'].fillna(-1, inplace=True)

trainItem.to_csv('../input/trainItem_w_features.csv')
testItem.to_csv('../input/testItem_w_features.csv')

trainItem.groupby('top_200_count1')['isDuplicate'].mean().plot()
plt.title('isDuplicate vs top_200_count1', fontsize='16')
plt.show()

trainItem.groupby('top_200_count2')['isDuplicate'].mean().plot()
plt.title('isDuplicate vs top_200_count2', fontsize='16')
plt.show()


trainItem['top_200_count_diff'] = np.abs(trainItem['top_200_count1'] - trainItem['top_200_count2'])
testItem['top_200_count_diff'] = np.abs(testItem['top_200_count1'] - testItem['top_200_count2'])

#trainItem.loc[trainItem['img_count_diff']>5, 'img_count_diff'] = 5
#testItem.loc[testItem['img_count_diff']>5, 'img_count_diff'] = 5

trainItem.groupby('top_200_count_diff')['isDuplicate'].mean().plot()
plt.title('isDuplicate vs top_200_count_diff', fontsize='16')
plt.show()


trainItem['top_200_count1'] = trainItem['top_200_count1'].astype(np.int16)
trainItem['top_200_count2'] = trainItem['top_200_count2'].astype(np.int16)
trainItem['top_200_count_diff'] = trainItem['top_200_count_diff'].astype(np.int16)

testItem['top_200_count1'] = testItem['top_200_count1'].astype(np.int16)
testItem['top_200_count2'] = testItem['top_200_count2'].astype(np.int16)
testItem['top_200_count_diff'] = testItem['top_200_count_diff'].astype(np.int16)

trainItem.to_csv('../input/trainItem_w_features.csv')
testItem.to_csv('../input/testItem_w_features.csv')





trainItem = pd.read_csv('../input/trainItem_w_features.csv')
trainItem.drop('Unnamed: 0', axis=1, inplace=True)
trainItem.to_csv('../input/trainItem_w_features.csv', index=False)



trainItem = pd.read_csv('../input/trainItem_w_features.csv')
testItem = pd.read_csv('../input/testItem_w_features.csv', index_col='id')




clusters_train = pd.read_csv('../input/clusters_train.csv')
clusters_train.drop('Unnamed: 0', axis=1, inplace=True)
clusters_test = pd.read_csv('../input/clusters_test.csv')
clusters_test.drop('Unnamed: 0', axis=1, inplace=True)


cluster_columns = ['cluster_x_15', 'cluster_y_15', 'same_cluster_15']
trainItem = pd.concat([trainItem, clusters_train[cluster_columns]], axis=1)
testItem = pd.concat([testItem, clusters_test[cluster_columns]], axis=1)




trainItem.to_csv('../input/trainItem_w_features.csv', index=False)
testItem.to_csv('../input/testItem_w_features.csv')








y = pd.DataFrame(trainItem.pop('isDuplicate'))

generation_method = pd.DataFrame(trainItem.pop('generationMethod'))

ItemIDs_train = trainItem[['itemID_1', 'itemID_2']].copy()
ItemIDs_test = testItem[['itemID_1', 'itemID_2']].copy()


trainItem.drop(['itemID_1', 'itemID_2'], axis=1, inplace=True)
testItem.drop(['itemID_1', 'itemID_2'], axis=1, inplace=True)





# swiping from:
# https://www.kaggle.com/dimon0981/avito-duplicate-ads-detection/python-xgboost-starter-0-77



from sklearn.cross_validation import train_test_split
import xgboost as xgb
import random
from operator import itemgetter
import zipfile
from sklearn.metrics import roc_auc_score
import time
random.seed(2016)

INPUT_DIR = '../input/'

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance


def intersect(a, b):
    return list(set(a) & set(b))


def print_features_importance(imp):
    for i in range(len(imp)):
        print("# " + str(imp[i][1]))
        print('output.remove(\'' + imp[i][0] + '\')')

























def run_default_test(train, test, features, target, random_state=0):
    eta = 0.1
    max_depth = 5
    subsample = 0.8
    colsample_bytree = 0.8
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state
    }
    num_boost_round = 260
    early_stopping_rounds = 20
    test_size = 0.1

    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_ntree_limit)
    score = roc_auc_score(X_valid[target].values, check)
    print('Check error value: {:.6f}'.format(score))

    imp = get_importance(gbm, features)
    print('Importance array: ', imp)

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_ntree_limit)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction.tolist(), score

def create_submission(score, test, prediction):
    # Make Submission
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('id,probability\n')
    total = 0
    for id in test['id']:
        str1 = str(id) + ',' + str(prediction[total])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()
    
    








'''

image_features_train = pd.read_csv('../input/train_dhash_counts.csv')
image_features_train.drop('Unnamed: 0', axis=1, inplace=True)
image_features_test = pd.read_csv('../input/test_dhash_counts.csv', index_col='id')

trainItem = pd.concat([trainItem, image_features_train], axis=1)
testItem = pd.concat([testItem, image_features_test], axis=1)


# groupby histgram















