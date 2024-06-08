from lyk_config import k, conf_th, DEBUG, load_data
import sys
sys.path.append('../input/timm045/')
import timm

from itertools import zip_longest
import json
import math
import gc
import os
from pathlib import Path

import faiss
import numpy as np
import cupy as cp
import pandas as pd

from tqdm import tqdm
from PIL import Image
import joblib
import lightgbm as lgb
from scipy.sparse import hstack, vstack, csc_matrix, csr_matrix
import editdistance
import networkx as nx

import string
import nltk
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

NUM_CLASSES = 11014
NUM_WORKERS = 2
SEED = 0

###
import time
from contextlib import contextmanager
from collections import defaultdict
map_used_time = defaultdict(float)
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    tt = time.time() - t0
    map_used_time[title] += tt
    print("  {} - done in {:.5f}s".format(title, tt))


df, img_dir = load_data()

stop_words = set([
    'promo','diskon','baik','terbaik', 'murah',
    'termurah', 'harga', 'price', 'best', 'seller',
    'bestseller', 'ready', 'stock', 'stok', 'limited',
    'bagus', 'kualitas', 'berkualitas', 'hari', 'ini',
    'jadi', 'gratis',
])

titles = [
    title.translate(str.maketrans({_: ' ' for _ in string.punctuation}))
    for title in df['title'].str.lower().values
]

tokenizer = TweetTokenizer()
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, 
                                   binary=True, 
                                   min_df=2, 
                                   token_pattern='(?u)\\b\\w+\\b', 
                                   tokenizer=tokenizer.tokenize,
                                   dtype=np.float32,
                                   norm='l2')
tfidf_feats = tfidf_vectorizer.fit_transform(titles)

with timer('load'):
    similarities_bert, indexes_bert = joblib.load('/tmp/lyk_bert_data.pkl')
    similarities_img, indexes_img = joblib.load('/tmp/lyk_img_data.pkl')
    st_sizes, img_hs, img_ws = joblib.load('/tmp/lyk_img_meta_data.pkl')
    similarities_mm, indexes_mm = joblib.load('/tmp/lyk_mm_data.pkl')
    
    row = indexes_bert.ravel()
    col = np.arange(len(indexes_bert)).repeat(k)
    data = similarities_bert.ravel()
    simmat_bert = {(i, j): d for i, j, d in zip(col, row, data)}

    row = indexes_img.ravel()
    col = np.arange(len(indexes_img)).repeat(k)
    data = similarities_img.ravel()
    simmat_img = {(i, j): d for i, j, d in zip(col, row, data)}

    row = indexes_mm.ravel()
    col = np.arange(len(indexes_mm)).repeat(k)
    data = similarities_mm.ravel()
    simmat_mm = {(i, j): d for i, j, d in zip(col, row, data)}

del row, col, data
gc.collect()

mean_sim_img_top5 = similarities_img[:, :5].mean(1)
mean_sim_bert_top5 = similarities_bert[:, :5].mean(1)
mean_mean_sim_img_top5 = mean_sim_img_top5[indexes_img[:, :5]].mean(1)
mean_mean_sim_bert_top5 = mean_sim_bert_top5[indexes_bert[:, :5]].mean(1)

mean_sim_img_top5 = (mean_sim_img_top5 - mean_sim_img_top5.mean()) / mean_sim_img_top5.std()
mean_sim_bert_top5 = (mean_sim_bert_top5 - mean_sim_bert_top5.mean()) / mean_sim_bert_top5.std()
mean_mean_sim_img_top5 = (mean_mean_sim_img_top5 - mean_mean_sim_img_top5.mean()) / mean_mean_sim_img_top5.std()
mean_mean_sim_bert_top5 = (mean_mean_sim_bert_top5 - mean_mean_sim_bert_top5.mean()) / mean_mean_sim_bert_top5.std()

mean_sim_img_top15 = similarities_img[:, :15].mean(1)
mean_sim_bert_top15 = similarities_bert[:, :15].mean(1)
mean_sim_img_top15 = (mean_sim_img_top15 - mean_sim_img_top15.mean()) / mean_sim_img_top15.std()
mean_sim_bert_top15 = (mean_sim_bert_top15 - mean_sim_bert_top15.mean()) / mean_sim_bert_top15.std()

mean_sim_img_top30 = similarities_img[:, :30].mean(1)
mean_sim_bert_top30 = similarities_bert[:, :30].mean(1)
mean_sim_img_top30 = (mean_sim_img_top30 - mean_sim_img_top30.mean()) / mean_sim_img_top30.std()
mean_sim_bert_top30 = (mean_sim_bert_top30 - mean_sim_bert_top30.mean()) / mean_sim_bert_top30.std()

mean_sim_mm_top5 = similarities_mm[:, :5].mean(1)
mean_mean_sim_mm_top5 = mean_sim_mm_top5[indexes_mm[:, :5]].mean(1)

mean_sim_mm_top5 = (mean_sim_mm_top5 - mean_sim_mm_top5.mean()) / mean_sim_mm_top5.std()
mean_mean_sim_mm_top5 = (mean_mean_sim_mm_top5 - mean_mean_sim_mm_top5.mean()) / mean_mean_sim_mm_top5.std()

mean_sim_mm_top15 = similarities_mm[:, :15].mean(1)
mean_sim_mm_top15 = (mean_sim_mm_top15 - mean_sim_mm_top15.mean()) / mean_sim_mm_top15.std()

mean_sim_mm_top30 = similarities_mm[:, :30].mean(1)
mean_sim_mm_top30 = (mean_sim_mm_top30 - mean_sim_mm_top30.mean()) / mean_sim_mm_top30.std()

row_titles = df['title'].values
posting_ids = df['posting_id'].values

tmp_dir = Path('/tmp/rows')
tmp_dir.mkdir(exist_ok=True, parents=True)

rows = []
for i in tqdm(range(len(df))):
    right_indexes = set(indexes_img[i].tolist() + indexes_bert[i].tolist())

    for _, j in enumerate(right_indexes):
        if i == j:
            continue
        sim_img = simmat_img.get((i, j), 0)
        sim_bert = simmat_bert.get((i, j), 0)
        sim_mm = simmat_mm.get((i, j), 0)
        if sim_img == 0 and sim_bert == 0:
            continue

        rows.append({
            'i': i,
            'j': j,
            'posting_id': posting_ids[i],
            'posting_id_target': posting_ids[j],
            'sim_img': sim_img,
            'sim_bert': sim_bert,
            'sim_mm': sim_mm,
            'edit_distance': editdistance.eval(titles[i], titles[j]),
            'title_len': len(row_titles[i]),
            'title_len_target': len(row_titles[j]),
            'title_num_words': len(row_titles[i].split()),
            'title_num_words_target': len(row_titles[j].split()),
            'mean_sim_img_top5': mean_sim_img_top5[i],
            'mean_sim_img_target_top5': mean_sim_img_top5[j],
            'mean_sim_bert_top5': mean_sim_bert_top5[i],
            'mean_sim_bert_target_top5': mean_sim_bert_top5[j],
            'mean_sim_img_top15': mean_sim_img_top15[i],
            'mean_sim_img_target_top15': mean_sim_img_top15[j],
            'mean_sim_bert_top15': mean_sim_bert_top15[i],
            'mean_sim_bert_target_top15': mean_sim_bert_top15[j],
            'mean_sim_img_top30': mean_sim_img_top30[i],
            'mean_sim_img_target_top30': mean_sim_img_top30[j],
            'mean_sim_bert_top30': mean_sim_bert_top30[i],
            'mean_sim_bert_target_top30': mean_sim_bert_top30[j],
            'st_size': st_sizes[i],
            'st_size_target': st_sizes[j],
            'wxh/st_size': img_ws[i] * img_hs[i] / st_sizes[i],
            'wxh/st_size_target': img_ws[j] * img_hs[j] / st_sizes[j],
            'mean_mean_sim_img_top5': mean_mean_sim_img_top5[i],
            'mean_mean_sim_img_target_top5': mean_mean_sim_img_top5[j],
            'mean_mean_sim_bert_top5': mean_mean_sim_bert_top5[i],
            'mean_mean_sim_bert_target_top5': mean_mean_sim_bert_top5[j],
            'mean_sim_mm_top5': mean_sim_mm_top5[i],
            'mean_sim_mm_target_top5': mean_sim_mm_top5[j],
            'mean_sim_mm_top15': mean_sim_mm_top15[i],
            'mean_sim_mm_target_top15': mean_sim_mm_top15[j],
            'mean_sim_mm_top30': mean_sim_mm_top30[i],
            'mean_sim_mm_target_top30': mean_sim_mm_top30[j],
            'mean_mean_sim_mm_top5': mean_mean_sim_mm_top5[i],
            'mean_mean_sim_mm_target_top5': mean_mean_sim_mm_top5[j],
        })

    if i % 10000 == 9999 or i == len(df) - 1:
        tmp_df = pd.DataFrame(rows)
        for col in tmp_df.columns:
            if tmp_df[col].dtype == 'float64':
                tmp_df[col] = tmp_df[col].astype('float32')
            elif tmp_df[col].dtype == 'int64':
                tmp_df[col] = tmp_df[col].astype('int32')
        tmp_df.to_feather(tmp_dir / f'{i}.feather')
        rows = []

df.drop(['image', 'title'], axis=1, inplace=True)
del (
    mean_sim_img_top5, mean_sim_img_top15, mean_sim_img_top30, mean_mean_sim_img_top5,
    mean_sim_bert_top5, mean_sim_bert_top15, mean_sim_bert_top30, mean_mean_sim_bert_top5,
    mean_sim_mm_top5, mean_sim_mm_top15, mean_sim_mm_top30, mean_mean_sim_mm_top5,
    simmat_img, simmat_bert, simmat_mm,
    similarities_img, indexes_img,
    similarities_bert, indexes_bert,
    similarities_mm, indexes_mm,
)
gc.collect()
with timer('to_frame'):
    df_pair = pd.concat([pd.read_feather(path) for path in tmp_dir.glob('**/*.feather')], axis=0).reset_index(drop=True)
del rows
gc.collect()

with timer('sim_tfidf'):
    df_pair['sim_tfidf'] = tfidf_feats[df_pair['i'].values].multiply(tfidf_feats[df_pair['j'].values]).sum(axis=1)
df_pair['title_len_diff'] = np.abs(df_pair['title_len'] - df_pair['title_len_target'])
df_pair['title_num_words_diff'] = np.abs(df_pair['title_num_words'] - df_pair['title_num_words_target'])

del tfidf_feats
gc.collect()
###

from cuml import ForestInference
import treelite
list_clf = []
for clf in joblib.load('../input/shopee/boosters_v34_v45_mm.pickle'):
    clf.save_model('/tmp/tmp.lgb')
    fi = ForestInference()
    fi.load_from_treelite_model(treelite.Model.load('/tmp/tmp.lgb', model_format='lightgbm'))
    list_clf.append(fi)

X = df_pair[[
    'sim_img', 'sim_tfidf', 'sim_bert', 'sim_mm', 'edit_distance',
    'title_len', 'title_len_target', 'title_len_diff',
    'title_num_words', 'title_num_words_target', 'title_num_words_diff',
    'mean_sim_img_top5', 'mean_sim_img_target_top5',
    'mean_sim_bert_top5', 'mean_sim_bert_target_top5',
    'mean_sim_mm_top5', 'mean_sim_mm_target_top5',
    'mean_sim_img_top15', 'mean_sim_img_target_top15',
    'mean_sim_bert_top15', 'mean_sim_bert_target_top15',
    'mean_sim_mm_top15', 'mean_sim_mm_target_top15',
    'mean_sim_img_top30', 'mean_sim_img_target_top30',
    'mean_sim_bert_top30', 'mean_sim_bert_target_top30',
    'mean_sim_mm_top30', 'mean_sim_mm_target_top30',
    'st_size', 'st_size_target',
    'wxh/st_size', 'wxh/st_size_target',
    'mean_mean_sim_img_top5', 'mean_mean_sim_img_target_top5',
    'mean_mean_sim_bert_top5', 'mean_mean_sim_bert_target_top5',
    'mean_mean_sim_mm_top5', 'mean_mean_sim_mm_target_top5',
]]

## passing as cupy array might be able to avoid multipy copy to GPU.
X = cp.asarray(X[clf.feature_name()].values.astype(np.float32))
df_pair = df_pair[['posting_id', 'posting_id_target']]

gc.collect()
with timer('predict'):
    df_pair['pred'] = np.mean([clf.predict(X).get() for clf in list_clf], axis=0) - conf_th

df_pair.to_pickle('submission_lyak.pkl')
