# -​*- coding: utf-8 -*​-

import pandas as pd
import numpy as np
import gensim
import sys

from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import *

import nltk
import time

stops = set(stopwords.words("russian"))
tokenizer = nltk.data.load('tokenizers/punkt/russian.pickle')

def text_to_list(text):
    words = tokenizer.tokenize(str(text).strip().lower())
    words = [w for w in words if not w in stops]
    print(words)
    return words

def prep_train():
    testing = 0
    start_time = time.time()

    types1 = {
        'itemID_1': np.dtype(int),
        'itemID_2': np.dtype(int),
        'isDuplicate': np.dtype(int),
        'generationMethod': np.dtype(int),
    }

    types2 = {
        'itemID': np.dtype(int),
        'categoryID': np.dtype(int),
        'title': np.dtype(str),
        'description': np.dtype(str),
        'images_array': np.dtype(str),
        'attrsJSON': np.dtype(str),
        'price': np.dtype(float),
        'locationID': np.dtype(int),
        'metroID': np.dtype(float),
        'lat': np.dtype(float),
        'lon': np.dtype(float),
    }

    print("Load ItemPairs_train.csv")
    pairs = pd.read_csv("../Data/ItemPairs_train.csv", dtype=types1)
    
    # Add 'id' column for easy merge
    print("Load ItemInfo_train.csv")
    items = pd.read_csv("../Data/ItemInfo_train.csv", dtype=types2)
    items.fillna(-1, inplace=True)

    train = pairs
    train = train.drop(['generationMethod'], axis=1)

    print('Merge item 1...')
    item1 = items[['itemID', 'title', 'description', 'attrsJSON']]

    item1 = item1.rename(
        columns={
            'itemID': 'itemID_1',
            'title' : 'title_1',
            'description' : 'description_1',
            'attrsJSON' : 'attrsJSON_1'
        }
    )

    # Add item 1 data
    train = pd.merge(train, item1, how='left', on='itemID_1', left_index=True)

    print('Merge item 2...')
    item2 = items[['itemID', 'title', 'description', 'attrsJSON']]

    item2 = item2.rename(
        columns = {
            'itemID': 'itemID_2',
            'title' : 'title_2',
            'description' : 'description_2',
            'attrsJSON' : 'attrsJSON_2'
        }
    )

    # Add item 2 data
    train = pd.merge(train, item2, how='left', on='itemID_2', left_index=True)

    print('Create train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return train

def prep_test():
    start_time = time.time()

    types1 = {
        'itemID_1': np.dtype(int),
        'itemID_2': np.dtype(int),
        'id': np.dtype(int),
    }

    types2 = {
        'itemID': np.dtype(int),
        'categoryID': np.dtype(int),
        'title': np.dtype(str),
        'description': np.dtype(str),
        'images_array': np.dtype(str),
        'attrsJSON': np.dtype(str),
        'price': np.dtype(float),
        'locationID': np.dtype(int),
        'metroID': np.dtype(float),
        'lat': np.dtype(float),
        'lon': np.dtype(float),
    }

    print("Load ItemPairs_test.csv")
    pairs = pd.read_csv("../Data/ItemPairs_test.csv", dtype=types1)
    print("Load ItemInfo_testcsv")
    items = pd.read_csv("../Data/ItemInfo_test.csv", dtype=types2)
    items.fillna(-1, inplace=True)
    train = pairs

    print('Merge item 1...')
    item1 = items[['itemID', 'title', 'description', 'attrsJSON']]
    
    item1 = item1.rename(
        columns={
            'itemID': 'itemID_1',
            'title' : 'title_1',
            'description' : 'description_1',
            'attrsJSON' : 'attrsJSON_1'
        }
    )

    # Add item 1 data
    train = pd.merge(train, item1, how='left', on='itemID_1', left_index=True)

    print('Merge item 2...')
    item2 = items[['itemID', 'title', 'description', 'attrsJSON']]
 
    item2 = item2.rename(
        columns={
            'itemID': 'itemID_2',
            'title' : 'title_2',
            'description' : 'description_2',
            'attrsJSON' : 'attrsJSON_2'
        }
    )

    # Add item 2 data
    train = pd.merge(train, item2, how='left', on='itemID_2', left_index=True)

    # print(train.describe())
    print('Create test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return train

if __name__ == '__main__':
    train = prep_train()

    train['title_1'] = train['title_1'].apply(lambda s : text_to_list(s))
    train['title_2'] = train['title_2'].apply(lambda s : text_to_list(s))
    train['description_1'] = train['description_1'].apply(lambda s : text_to_list(s))
    train['description_2'] = train['description_2'].apply(lambda s : text_to_list(s))
    train['attrsJSON_1'] = train['attrsJSON_1'].apply(lambda s : text_to_list(s))
    train['attrsJSON_2'] = train['attrsJSON_2'].apply(lambda s : text_to_list(s))
    
    test = prep_train()
    test['title_1'] = test['title_1'].apply(lambda s : text_to_list(s))
    test['title_2'] = test['title_2'].apply(lambda s : text_to_list(s))
    test['description_1'] = test['description_1'].apply(lambda s : text_to_list(s))
    test['description_2'] = test['description_2'].apply(lambda s : text_to_list(s))
    test['attrsJSON_1'] = test['attrsJSON_1'].apply(lambda s : text_to_list(s))
    test['attrsJSON_2'] = test['attrsJSON_2'].apply(lambda s : text_to_list(s))
    
    print("Loading word2vec model1")

    model = Word2Vec.load_word2vec_format('../PretrainedModels/ruscorpora.model.bin', binary=True)
    
    print("Computing feature for model1")

    train['title_sim_word2vec1'] = train.apply(lambda row: model.n_similarity(row['title_1'], row['title_2']), axis=1)
    train['description_sim_word2vec1'] = train.apply(lambda row: model.n_similarity(row['description_1'], row['description_2']), axis=1)
    train['attrsJSON_sim_word2vec1'] = train.apply(lambda row: model.n_similarity(row['attrsJSON_1'], row['attrsJSON_2']), axis=1)
    
    test['title_sim_word2vec1'] = test.apply(lambda row: model.n_similarity(row['title_1'], row['title_2']), axis=1)
    test['description_sim_word2vec1'] = test.apply(lambda row: model.n_similarity(row['description_1'], row['description_2']), axis=1)
    test['attrsJSON_sim_word2vec1'] = test.apply(lambda row: model.n_similarity(row['attrsJSON_1'], row['attrsJSON_2']), axis=1)
    
    print("Loading word2vec model2")

    model = Word2Vec.load_word2vec_format('../PretrainedModels/web_russe.model.bin', binary=True)
  
    print("Computing feature for model2")

    train['title_sim_word2vec2'] = train.apply(lambda row: model.n_similarity(row['title_1'], row['title_2']), axis=1)
    train['description_sim_word2vec2'] = train.apply(lambda row: model.n_similarity(row['description_1'], row['description_2']), axis=1)
    train['attrsJSON_sim_word2vec2'] = train.apply(lambda row: model.n_similarity(row['attrsJSON_1'], row['attrsJSON_2']), axis=1)
    
    test['title_sim_word2vec2'] = test.apply(lambda row: model.n_similarity(row['title_1'], row['title_2']), axis=1)
    test['description_sim_word2vec2'] = test.apply(lambda row: model.n_similarity(row['description_1'], row['description_2']), axis=1)
    test['attrsJSON_sim_word2vec2'] = test.apply(lambda row: model.n_similarity(row['attrsJSON_1'], row['attrsJSON_2']), axis=1)
   
    print("Loading word2vec model3")

    model = Word2Vec.load_word2vec_format('../PretrainedModels/news_russe.model.bin', binary=True)
  
    print("Computing feature for model3")

    train['title_sim_word2vec3'] = train.apply(lambda row: model.n_similarity(row['title_1'], row['title_2']), axis=1)
    train['description_sim_word2vec3'] = train.apply(lambda row: model.n_similarity(row['description_1'], row['description_2']), axis=1)
    train['attrsJSON_sim_word2vec3'] = train.apply(lambda row: model.n_similarity(row['attrsJSON_1'], row['attrsJSON_2']), axis=1)
    
    test['title_sim_word2vec3'] = test.apply(lambda row: model.n_similarity(row['title_1'], row['title_2']), axis=1)
    test['description_sim_word2vec3'] = test.apply(lambda row: model.n_similarity(row['description_1'], row['description_2']), axis=1)
    test['attrsJSON_sim_word2vec3'] = test.apply(lambda row: model.n_similarity(row['attrsJSON_1'], row['attrsJSON_2']), axis=1)
 
    print("Loading word2vec model4")

    model = Word2Vec.load_word2vec_format('../PretrainedModels/ruwikiruscorpora.model.bin', binary=True)
  
    print("Computing feature for model4")

    train['title_sim_word2vec4'] = train.apply(lambda row: model.n_similarity(row['title_1'], row['title_2']), axis=1)
    train['description_sim_word2vec4'] = train.apply(lambda row: model.n_similarity(row['description_1'], row['description_2']), axis=1)
    train['attrsJSON_sim_word2vec4'] = train.apply(lambda row: model.n_similarity(row['attrsJSON_1'], row['attrsJSON_2']), axis=1)
    
    test['title_sim_word2vec4'] = test.apply(lambda row: model.n_similarity(row['title_1'], row['title_2']), axis=1)
    test['description_sim_word2vec4'] = test.apply(lambda row: model.n_similarity(row['description_1'], row['description_2']), axis=1)
    test['attrsJSON_sim_word2vec4'] = test.apply(lambda row: model.n_similarity(row['attrsJSON_1'], row['attrsJSON_2']), axis=1)
 

    features = ['itemID_1', 'itemID_2', 'title_sim_word2vec1', 'title_sim_word2vec2', 'title_sim_word2vec3', 'title_sim_word2vec4', \
    'description_sim_word2vec1', 'description_sim_word2vec2', 'description_sim_word2vec3', 'description_sim_word2vec4', \
    'attrsJSON_sim_word2vec1', 'attrsJSON_sim_word2vec2', 'attrsJSON_sim_word2vec3', 'attrsJSON_sim_word2vec4']

    save_train = train[features]
    save_test = test[features]

    save_train.to_csv("../features/word2vec_features_train.csv", sep=',')
    save_test.to_csv("../features/word2vec_features_test.csv", sep=',')




