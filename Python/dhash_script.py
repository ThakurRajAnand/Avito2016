# -*- coding: utf-8 -*-
"""
https://www.kaggle.com/rightfit/avito-duplicate-ads-detection/get-hash-from-images/code
"""


import numpy as np
import pandas as pd


import zipfile
import os
import io
from PIL import Image
#os.chdir('/home/run2/avito')
os.chdir('../input')
#import pandas as pd
import datetime

import feather

def dhash(image,hash_size = 16):
    image = image.convert('LA').resize((hash_size+1,hash_size),Image.ANTIALIAS)
    difference = []
    for row in xrange(hash_size):
        for col in xrange(hash_size):
            pixel_left = image.getpixel((col,row))
            pixel_right = image.getpixel((col+1,row))
            difference.append(pixel_left>pixel_right)
    decimal_value = 0
    hex_string = []
    for index, value in enumerate(difference):
        if value:
            decimal_value += 2**(index%8)
        if (index%8) == 7:
            hex_string.append(hex(decimal_value)[2:].rjust(2,'0'))
            decimal_value = 0

    return ''.join(hex_string)


image_features = feather.read_dataframe('../input/image_features.feather')

def image_path(x):
    return '../input/images/{}/{}.jpg'.format('{:02d}'.format(int(x[-2:])), str(x))








image_features['image_file'] = image_features['image'].apply(lambda x: image_path(x))

image_features['dhash'] = np.nan

counter = 0
for idx in image_features.index:

    img_file = image_features.loc[idx, 'image_file']

    try:
        img = Image.open(img_file)
        img_hash = dhash(img)
        image_features.loc[idx, 'dhash'] = img_hash
        counter+=1
        if counter%10000==0:
            print 'Done ' + str(counter) , datetime.datetime.now()

    except:
        print ('Could not read ' + str(img_file) )


image_features.drop('image_file', axis=1, inplace=True)

feather.write_dataframe(image_features, '../input/image_features_dhash.feather')

image_features.to_csv('../input/image_features_dhash.csv', index=False)




#### Flip image

image_features = feather.read_dataframe('../input/image_features.feather')

image_features['image_file'] = image_features['image'].apply(lambda x: image_path(x))

image_features['dhash_flip'] = np.nan

counter = 0
for idx in image_features.index:

    img_file = image_features.loc[idx, 'image_file']

    try:
        img = Image.open(img_file).transpose(Image.FLIP_LEFT_RIGHT)
        img_hash = dhash(img)
        image_features.loc[idx, 'dhash_flip'] = img_hash
        counter+=1
        if counter%10000==0:
            print 'Done ' + str(counter) , datetime.datetime.now()

    except:
        print ('Could not read ' + str(img_file) )


image_features.drop('image_file', axis=1, inplace=True)

feather.write_dataframe(image_features, '../input/image_features_dhash_flip.feather')

image_features.to_csv('../input/image_features_dhash_flip.csv', index=False)



