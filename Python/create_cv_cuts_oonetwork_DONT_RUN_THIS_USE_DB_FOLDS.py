import pandas as pd
from sklearn.cross_validation import LabelKFold

train = pd.read_csv('item_network_id_train.csv')
train.head()

skf = LabelKFold(train.network_id, n_folds=5)
cvs = []
for k, (train_index, valid_index) in enumerate(skf, 1):
    df = train.iloc[valid_index]
    df['k_fold'] = k
    cvs.append(df)
cvs = pd.concat(cvs, ignore_index=True)

train = pd.read_csv('../input/ItemPairs_train.csv')

cvs = cvs.merge(train, how='inner', on=['itemID_1', 'itemID_2'])

print cvs.groupby('k_fold')['isDuplicate'].aggregate(['count', 'mean'])

# check no overlap
for k in range(1, 6):
    in_fold = set(cvs[cvs.k_fold == k].network_id.values)
    out_fold = set(cvs[cvs.k_fold != k].network_id.values)
    print 'K: {:}; Number network_id overlap {:,}'.format(k, len(in_fold & out_fold))

cvs[['itemID_1', 'itemID_2', 'k_fold']].to_csv('train_cv_folds_oonetwork.csv', index=False)
