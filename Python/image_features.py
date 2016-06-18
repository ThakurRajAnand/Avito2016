# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
from scipy import spatial
import pandas as pd
import cv2

import feather

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD



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
        
        
        
'''
def get_image_names(directory):

    image_names = []

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            image_names.append(filename[:-4])  # all but last 4 chars

    return image_names


image_names = get_image_names(r'../input/images')
image_features = pd.DataFrame(columns=['image'], data=image_names)

feather.write_dataframe(image_features, '../input/image_features.feather')
'''






def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


model = VGG_16('../input/vgg16_weights.h5')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')


vgg16_cols = ['vgg16_{}'.format(i) for i in range(1000)]


image_features = feather.read_dataframe('../input/image_features.feather')
image_features = image_features[image_features['image'].apply(lambda x: x[-2:-1])=='9']
preds = np.zeros((image_features.index.shape[0], 1000), dtype=np.float32)

idx = 0
timer = ExecutionTime()

for i in image_features.index:
    try:
        img_num = image_features.loc[i, 'image']
        img_path = '../input/images/{}/{}.jpg'.format(img_num[-2:], img_num)
        im = cv2.resize(cv2.imread(img_path), (224, 224)).astype(np.float32)
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        im = im.transpose((2,0,1))
        im = np.expand_dims(im, axis=0)
        
        out = model.predict(im)
        preds[idx,:] = out.ravel()
        
        idx += 1
        if idx % 10000 == 0:
            print('finished {:6d} in {:0.2f} seconds'.format(idx, timer.duration()))
            timer = ExecutionTime()
    except:
        print('Puked on image: {}'.format(img_num))  
        
        
### showing classes above 0.1
        
preds = preds.astype(np.float64)
vgg16 = pd.DataFrame(index=image_features.index, columns=vgg16_cols, data=preds)
image_features = pd.concat([image_features, vgg16], axis=1)
feather.write_dataframe(image_features, '../input/image_features_9_vgg16.feather')



features = feather.read_dataframe('../input/image_features_0_vgg16.feather')
features = features.set_index('image', drop=True)
vgg16_gt_010 = features.apply(lambda x: np.where(x>=0.1), axis=1).apply(lambda x: x[0])

for i in range(1,10):

    features = feather.read_dataframe('../input/image_features_{}_vgg16.feather'.format(i))
    features = features.set_index('image', drop=True)
    new = features.apply(lambda x: np.where(x>=0.1), axis=1).apply(lambda x: x[0])
    vgg16_gt_010 = pd.concat([vgg16_gt_010, new], axis=0)


vgg16_gt_010 = pd.DataFrame(vgg16_gt_010)
vgg16_gt_010 = vgg16_gt_010.rename(columns={0: 'categories'})
vgg16_gt_010.to_csv('../input/vgg16_gt_010.csv')



#####


features = feather.read_dataframe('../input/image_features_0_vgg16.feather')
features = features.set_index('image', drop=True)

f16_0 = features.astype(np.float16)

features = feather.read_dataframe('../input/image_features_1_vgg16.feather')
features = features.set_index('image', drop=True)

f16_1 = features.astype(np.float16)

features = feather.read_dataframe('../input/image_features_2_vgg16.feather')
features = features.set_index('image', drop=True)

f16_2 = features.astype(np.float16)

features = feather.read_dataframe('../input/image_features_3_vgg16.feather')
features = features.set_index('image', drop=True)

f16_3 = features.astype(np.float16)

features = feather.read_dataframe('../input/image_features_4_vgg16.feather')
features = features.set_index('image', drop=True)

f16_4 = features.astype(np.float16)

features = feather.read_dataframe('../input/image_features_5_vgg16.feather')
features = features.set_index('image', drop=True)

f16_5 = features.astype(np.float16)

features = feather.read_dataframe('../input/image_features_6_vgg16.feather')
features = features.set_index('image', drop=True)

f16_6 = features.astype(np.float16)

features = feather.read_dataframe('../input/image_features_7_vgg16.feather')
features = features.set_index('image', drop=True)

f16_7 = features.astype(np.float16)

features = feather.read_dataframe('../input/image_features_8_vgg16.feather')
features = features.set_index('image', drop=True)

f16_8 = features.astype(np.float16)

features = feather.read_dataframe('../input/image_features_9_vgg16.feather')
features = features.set_index('image', drop=True)

f16_9 = features.astype(np.float16)

f16 = pd.concat([f16_0, f16_1, f16_2, f16_3, f16_4, f16_5, f16_6, f16_7, f16_8, f16_9])

f16.to_csv('/media/walter/data_science/kaggle/avito_overflow/image_features_vgg16.csv')

del f16_0,f16_1,f16_2,f16_3,f16_4,f16_5,f16_6,f16_7,f16_8,f16_9,features
gc.collect() 







def euc_func(px, py):
    try:
        px = px.split(',')
        px = [i.strip() for i in px]
        py = py.split(',')
        py = [i.strip() for i in py]
        if px and py:
            euc_dist = []
            for x in px:
                for y in py:
                    x_vgg = f16.loc[x].values
                    y_vgg = f16.loc[y].values
                    euc = spatial.distance.euclidean(x_vgg, y_vgg)
                    euc_dist.append(euc)
            return np.percentile(euc_dist, 25)
    except:
        return np.nan


info = pd.read_csv('../input/ItemInfo_train.csv')
pairs = pd.read_csv('../input/ItemPairs_train.csv')

info = info.drop(['title','description','attrsJSON'], axis=1)

pairs = pd.merge(pd.merge(pairs, info, how='inner', left_on='itemID_1', right_on='itemID'), 
              info, how='inner', left_on='itemID_2', right_on='itemID')

pairs['images_array_x'].fillna('', inplace=True)
pairs['images_array_y'].fillna('', inplace=True)

pairs['euc_dist_25'] = np.nan


timer = ExecutionTime()

pairs['euc_dist_25'] = \
    pairs[['images_array_x', 'images_array_y']].apply(lambda x: euc_func(x[0], x[1]), axis=1)

print('finished in {:0.2f} minutes'.format(timer.duration()/60))


vgg_euc_train = pd.DataFrame(pairs['euc_dist_25'])
vgg_euc_train.to_csv('../input/vcc_euc_train.csv')




info = pd.read_csv('../input/ItemInfo_test.csv')
pairs = pd.read_csv('../input/ItemPairs_test.csv')
info = info.drop(['title','description','attrsJSON'], axis=1)
pairs = pd.merge(pd.merge(pairs, info, how='inner', left_on='itemID_1', right_on='itemID'), 
              info, how='inner', left_on='itemID_2', right_on='itemID')

pairs['images_array_x'].fillna('', inplace=True)
pairs['images_array_y'].fillna('', inplace=True)
pairs['euc_dist_25'] = np.nan


timer = ExecutionTime()

pairs['euc_dist_25'] = \
    pairs[['images_array_x', 'images_array_y']].apply(lambda x: euc_func(x[0], x[1]), axis=1)

print('finished in {:0.2f} minutes'.format(timer.duration()/60))

vgg_euc_test = pd.DataFrame(pairs['euc_dist_25'])
vgg_euc_test.to_csv('../input/vcc_euc_test.csv')





'''








    
   #100 per 1000 


plt.hist(pairs['euc_dist_25'].head(1000)[pairs['isDuplicate']==0], bins=20, alpha=0.7, label='not dup')
plt.hist(pairs['euc_dist_25'].head(1000)[pairs['isDuplicate']==1], bins=20, alpha=0.7, label='dup')
plt.legend()
plt.show()


'''


# continuing

vgg16_cols = ['vgg16_{}'.format(i) for i in range(1000)]

image_features = feather.read_dataframe('../input/image_features_2_vgg16.feather')

subset = image_features[image_features[vgg16_cols].sum(1)<0.5].index.tolist()

preds = np.zeros((len(subset), 1000), dtype=np.float32)


idx = 0
timer = ExecutionTime()

for i in subset:
    
    try:
        img_num = image_features.loc[i, 'image']
        img_path = '../input/images/{}/{}.jpg'.format(img_num[-2:], img_num)
        im = cv2.resize(cv2.imread(img_path), (224, 224)).astype(np.float32)
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        im = im.transpose((2,0,1))
        im = np.expand_dims(im, axis=0)
        
        out = model.predict(im)
        preds[idx,:] = out.ravel()
    except:
        print('Puked on image: {}'.format(img_num))
    
    idx += 1
    if idx % 10000 == 0:
        print('finished {:6d} in {:0.2f} seconds'.format(idx, timer.duration()))
        timer = ExecutionTime()



preds = preds.astype(np.float64)
image_features.loc[subset, vgg16_cols] = preds
feather.write_dataframe(image_features, '../input/image_features_2_vgg16.feather')






# 9268406
# 4515613
# 9322814

opencv-2.4.10/modules/imgproc/src/imgwarp.cpp:1968: error: (-215) ssize.area() > 0 in function resize

'''
