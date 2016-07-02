# coding: utf-8
import pandas as pd
cvs = pd.read_csv('train_cv_folds_oonetwork.csv')
cvs.info()

train = pd.read_csv('../input/ItemPairs_train.csv', usecols=['itemID_1', 'itemID_2', 'isDuplicate']).merge(cvs, how='inner', on=['itemID_1', 'itemID_2'])
test = pd.read_csv('../input/ItemPairs_test.csv')
print train.info()
print test.info()

train_info = pd.read_csv('../input/ItemInfo_train.csv', usecols=['itemID', 'categoryID'])
test_info = pd.read_csv('../input/ItemInfo_test.csv', usecols=['itemID', 'categoryID'])
print train_info.info()
print test_info.info()

train = train.merge(train_info, how='left', left_on='itemID_1', right_on='itemID').drop('itemID', 1)
test = test.merge(test_info, how='left', left_on='itemID_1', right_on='itemID').drop('itemID', 1)

train.head()

combined = []
for k in range(1, 6):
    oof = train[train.k_fold != k][['categoryID', 'isDuplicate']]
    infold = train[train.k_fold == k][['itemID_1', 'itemID_2', 'categoryID']]
    oof = oof.groupby('categoryID')['isDuplicate'].aggregate(['count', 'mean']).reset_index()
    oof.columns = ['categoryID', 'categoryIDCount', 'categoryIDTargetMean']
    infold = infold.merge(oof, how='left', on='categoryID')
    combined.append(infold)
combined = pd.concat(combined, ignore_index=True)

combined.info()

train_means = train.groupby('categoryID').isDuplicate.aggregate(['count', 'mean']).reset_index()
train_means.columns = ['categoryID', 'categoryIDCount', 'categoryIDTargetMean']
test = test.merge(train_means, how='left', on='categoryID')

test.info()

combined.to_csv('features_categoryIDtargetmeansoof_train.csv', index=False)

test.to_csv('features_categoryIDtargetmeansoof_test.csv', index=False)
