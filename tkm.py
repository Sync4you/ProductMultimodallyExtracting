import pandas as pd
import numpy as np
import sys
import ast
import os
import time
import cv2
import PIL.Image
import random
import joblib

from multiprocessing import Pool
from sklearn.metrics import accuracy_score

import langid
import Levenshtein

#import albumentations
#from albumentations import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
from sklearn.metrics import roc_auc_score
from warnings import filterwarnings

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


filterwarnings("ignore")


###

import imagesize
import Stemmer
stemmer = Stemmer.Stemmer('indonesian')
DEBUG = len(pd.read_csv('../input/shopee-product-matching/test.csv')) == 3


if DEBUG:
    data_dir = '../input/shopee-product-matching/train_images/'
else:
    data_dir = '../input/shopee-product-matching/test_images/'
    
###

if DEBUG:
    if 1:
        nrows = 1000
        df_test = pd.read_csv('../input/shopee-libs/train_newfold_stmmedid.csv', nrows=nrows)
    else:
        df_test = pd.read_csv('../input/shopee-libs/train_newfold_stmmedid.csv').append(
            pd.read_csv('../input/shopee-libs/train_newfold_stmmedid.csv'), ignore_index=True
        )
    
    label_groups = np.sort(df_test['label_group'].unique())
    map_label2id = {g: i for i, g in enumerate(label_groups)}
    df_test['label'] = df_test['label_group'].map(map_label2id)
    df_test['file_path'] = df_test.image.apply(lambda x: os.path.join(data_dir, f'{x}'))
else:
    df_test = pd.read_csv('../input/shopee-product-matching/test.csv')
    df_test['file_path'] = df_test.image.apply(lambda x: os.path.join(data_dir, f'{x}'))

    titles = df_test['title'].str.lower().values

    with timer('get lang'):
        df_test['lang'] = [langid.classify(t)[0] for t in tqdm(titles)]
        list_lang = df_test['lang'].values
    with timer('lemmatize'):
        titles = np.array([t.encode('ascii').decode('unicode-escape').encode('ascii', 'replace').decode('ascii').replace('?', ' ') for t in titles])
        titles = [' '.join(stemmer.stemWords(t.split())) if list_lang[i] in {'id', 'ms'} else t for i, t in enumerate(tqdm(titles))]
        df_test['title'] = titles

with timer('get image size'):
    st_sizes, img_hs, img_ws = joblib.load('/tmp/lyk_img_meta_data.pkl')
    df_test['width'] = img_ws
    df_test['hight'] = img_hs
    df_test['st_size'] = st_sizes
    df_test['wxh/st_size'] = df_test['width'] * df_test['hight'] / df_test['st_size']

df_test.to_pickle('/tmp/df_test_tkm.pkl')
###

K = min(60, df_test.shape[0])

###
print('Computing text embeddings...')
import cupy as cp
import pickle
import gc
from cuml.feature_extraction.text import TfidfVectorizer
import cudf

model = TfidfVectorizer(stop_words=None, 
                        binary=True, 
                        max_features=100000,
                        max_df=0.3,
                        min_df=2,
                        dtype=np.float32)

with timer('tfidf fit'):
    titles = pd.read_csv('../input/shopee-libs/train_newfold_stmmedid.csv', 
                         usecols=['title'])['title'].values.tolist()
    test_titles = df_test.title.values.tolist()
    titles += test_titles
    model.fit(cudf.Series(titles))
    text_embeddings = model.transform(cudf.Series(test_titles))
    print('text embeddings shape',text_embeddings.shape)

with timer('tfidf pred'):
    CHUNK = 1024*4
    print('Finding similar titles...')
    text_D = np.zeros((df_test.shape[0], K), dtype=np.float32)
    text_I = np.zeros((df_test.shape[0], K), dtype=np.int32)


    CTS = text_embeddings.shape[0]//CHUNK
    if  text_embeddings.shape[0]%CHUNK!=0: CTS += 1
    cnt = 0
    for j in range( CTS ):

        a = j*CHUNK
        b = (j+1)*CHUNK
        b = min(b, text_embeddings.shape[0])
        print('chunk',a,'to',b, text_embeddings.shape[0])

        #COSINE SIMILARITY DISTANCE
        cts = (text_embeddings * text_embeddings[a:b].T).T.toarray()
        indices = cp.argsort(cts, axis=1)

        for k in range(b-a):
            idx = indices[k][::-1]
            text_I[cnt] = idx[:K].get()
            text_D[cnt] = cts[k, idx[:K]].get()
            cnt += 1

del text_embeddings, indices, cts
gc.collect()
###

img_D = np.load('/tmp/img_D_qe.npy')
img_I = np.load('/tmp/img_I_qe.npy')

###

bert_D = np.load('/tmp/brt_D_qe.npy')
bert_I = np.load('/tmp/brt_I_qe.npy')

###

bth_D = np.load('/tmp/bth_D_qe.npy')
bth_I = np.load('/tmp/bth_I_qe.npy')
###

mut_D = np.load('/tmp/mut_D_qe.npy')
mut_I = np.load('/tmp/mut_I_qe.npy')
###

map_col2id = {}
###

import langid
import Levenshtein
titles = df_test['title'].values
titles_set = [set(t) for t in titles]
langs = df_test['lang'].values
st_size = df_test['st_size'].values
wh_st_size = df_test['wxh/st_size'].values
###

numset = set('0123456789')

###
text_D = np.array(text_D)
txt_cnt_all = np.vstack([(text_D > t).sum(axis=1) for t in [0.9, 0.8, 0.7, 0.6, 0.5]]).T
txt_avg_raw_all = text_D.mean(axis=1)
txt_avg_all = (txt_avg_raw_all - txt_avg_raw_all.mean()) / txt_avg_raw_all.std()
txt_std_all = text_D.std(axis=1)

txt_avg_5_all = text_D[:, :5].mean(axis=1)
txt_avg_10_all = text_D[:, :10].mean(axis=1)
txt_avg_15_all = text_D[:, :15].mean(axis=1)
txt_avg_30_all = text_D[:, :30].mean(axis=1)

txt_avg_5_all = (txt_avg_5_all - txt_avg_5_all.mean()) / txt_avg_5_all.std()
txt_avg_10_all = (txt_avg_10_all - txt_avg_10_all.mean()) / txt_avg_10_all.std()
txt_avg_15_all = (txt_avg_15_all - txt_avg_15_all.mean()) / txt_avg_15_all.std()
txt_avg_30_all = (txt_avg_30_all - txt_avg_30_all.mean()) / txt_avg_30_all.std()
    
###
brt_cnt_all = np.vstack([(bert_D > t).sum(axis=1) for t in [0.9, 0.8, 0.7, 0.6, 0.5]]).T
brt_avg_raw_all = bert_D.mean(axis=1)
brt_avg_all = (brt_avg_raw_all - brt_avg_raw_all.mean()) / brt_avg_raw_all.std()
brt_std_all = bert_D.std(axis=1)

brt_avg_5_all = bert_D[:, :5].mean(axis=1)
brt_avg_10_all = bert_D[:, :10].mean(axis=1)
brt_avg_15_all = bert_D[:, :15].mean(axis=1)
brt_avg_30_all = bert_D[:, :30].mean(axis=1)

brt_avg_5_all = (brt_avg_5_all - brt_avg_5_all.mean()) / brt_avg_5_all.std()
brt_avg_10_all = (brt_avg_10_all - brt_avg_10_all.mean()) / brt_avg_10_all.std()
brt_avg_15_all = (brt_avg_15_all - brt_avg_15_all.mean()) / brt_avg_15_all.std()
brt_avg_30_all = (brt_avg_30_all - brt_avg_30_all.mean()) / brt_avg_30_all.std()

###
bth_cnt_all = np.vstack([(bth_D > t).sum(axis=1) for t in [0.9, 0.8, 0.7, 0.6, 0.5]]).T
bth_avg_raw_all = bth_D.mean(axis=1)
bth_avg_all = (bth_avg_raw_all - bth_avg_raw_all.mean()) / bth_avg_raw_all.std()
bth_std_all = bth_D.std(axis=1)

bth_avg_5_all = bth_D[:, :5].mean(axis=1)
bth_avg_10_all = bth_D[:, :10].mean(axis=1)
bth_avg_15_all = bth_D[:, :15].mean(axis=1)
bth_avg_30_all = bth_D[:, :30].mean(axis=1)

bth_avg_5_all = (bth_avg_5_all - bth_avg_5_all.mean()) / bth_avg_5_all.std()
bth_avg_10_all = (bth_avg_10_all - bth_avg_10_all.mean()) / bth_avg_10_all.std()
bth_avg_15_all = (bth_avg_15_all - bth_avg_15_all.mean()) / bth_avg_15_all.std()
bth_avg_30_all = (bth_avg_30_all - bth_avg_30_all.mean()) / bth_avg_30_all.std()
        
###
mut_cnt_all = np.vstack([(mut_D > t).sum(axis=1) for t in [0.9, 0.8, 0.7, 0.6, 0.5]]).T
mut_avg_raw_all = mut_D.mean(axis=1)
mut_avg_all = (mut_avg_raw_all - mut_avg_raw_all.mean()) / mut_avg_raw_all.std()
mut_std_all = mut_D.std(axis=1)

mut_avg_5_all = mut_D[:, :5].mean(axis=1)
mut_avg_10_all = mut_D[:, :10].mean(axis=1)
mut_avg_15_all = mut_D[:, :15].mean(axis=1)
mut_avg_30_all = mut_D[:, :30].mean(axis=1)

mut_avg_5_all = (mut_avg_5_all - mut_avg_5_all.mean()) / mut_avg_5_all.std()
mut_avg_10_all = (mut_avg_10_all - mut_avg_10_all.mean()) / mut_avg_10_all.std()
mut_avg_15_all = (mut_avg_15_all - mut_avg_15_all.mean()) / mut_avg_15_all.std()
mut_avg_30_all = (mut_avg_30_all - mut_avg_30_all.mean()) / mut_avg_30_all.std()
        
###
img_cnt_all = np.vstack([(img_D > t).sum(axis=1) for t in [0.9, 0.8, 0.7, 0.6, 0.5]]).T
img_avg_raw_all = img_D.mean(axis=1)
img_avg_all = (img_avg_raw_all - img_avg_raw_all.mean()) / img_avg_raw_all.std()
img_std_all = img_D.std(axis=1)

img_avg_5_all = img_D[:, :5].mean(axis=1)
img_avg_10_all = img_D[:, :10].mean(axis=1)
img_avg_15_all = img_D[:, :15].mean(axis=1)
img_avg_30_all = img_D[:, :30].mean(axis=1)

img_avg_5_all = (img_avg_5_all - img_avg_5_all.mean()) / img_avg_5_all.std()
img_avg_10_all = (img_avg_10_all - img_avg_10_all.mean()) / img_avg_10_all.std()
img_avg_15_all = (img_avg_15_all - img_avg_15_all.mean()) / img_avg_15_all.std()
img_avg_30_all = (img_avg_30_all - img_avg_30_all.mean()) / img_avg_30_all.std()

width_hight = df_test[['width', 'hight']].values

list_pred_id = [[] for _ in range(df_test.shape[0])]

indices = df_test.index.values

ptr = 0
all_feat = np.memmap('/tmp/tkm_feat.dat', dtype='float32', mode='w+', shape=(df_test.shape[0] * 60 * 5, 150), order='F')

feat = np.zeros((60 * 5, 150), dtype='float32')

list_idx = []
list_idx2 = []
list_feats = []
for i in tqdm(indices):
    img_d = img_D[i]
    img_i = img_I[i]

    img_cnt = img_cnt_all[i]
    img_avg = img_avg_all[i]
    img_std = img_std_all[i]

    img_width ,img_hight = width_hight[i]

    ###
    txt_d = text_D[i]
    txt_i = text_I[i]

    txt_cnt = txt_cnt_all[i]
    txt_avg = txt_avg_all[i]
    txt_std = txt_std_all[i]

    txt_set = set(titles[i])
    ###
    brt_d = bert_D[i]
    brt_i = bert_I[i]

    brt_cnt = brt_cnt_all[i]
    brt_avg = brt_avg_all[i]
    brt_std = brt_std_all[i]

    brt_set = set(titles[i])
    bth_d = bth_D[i]
    bth_i = bth_I[i]

    bth_cnt = bth_cnt_all[i]
    bth_avg = bth_avg_all[i]
    bth_std = bth_std_all[i]

    bth_set = set(titles[i])
    mut_d = mut_D[i]
    mut_i = mut_I[i]

    mut_cnt = mut_cnt_all[i]
    mut_avg = mut_avg_all[i]
    mut_std = mut_std_all[i]

    mut_set = set(titles[i])

    map_feat = {}
    for j in range(K):
        _w, _h = width_hight[img_i[j]]
        _img_cnt = img_cnt_all[img_i[j]]
        _img_avg = img_avg_all[img_i[j]]
        _img_std = img_std_all[img_i[j]]

        diff_width = abs(img_width - _w)
        diff_hight = abs(img_hight - _h)
        d = {
            'img_sim': img_d[j],
            'img_avg': img_avg, 
            'img_std': img_std,
            'img_avg2': _img_avg, 
            'img_std2': _img_std,

            'img_avg_raw': img_avg_raw_all[i],
            'img_avg2_raw': img_avg_raw_all[img_i[j]],

            'diff_width': diff_width,
            'diff_hight': diff_hight,
            'img_width': img_width,
            'img_hight': img_hight,
            'img_width2': _w,
            'img_hight2': _h,

            'st_size': st_size[i],
            'st_size2': st_size[img_i[j]],
            'wh_st_size': wh_st_size[i],
            'wh_st_size2': wh_st_size[img_i[j]]
        }
        d.update({f'img_cnt_{ii}': img_cnt[ii] for ii in range(img_cnt.shape[0])})
        d.update({f'img_cnt2_{ii}': _img_cnt[ii] for ii in range(_img_cnt.shape[0])})
        map_feat[img_i[j]] = d
        
    for j in range(K):
        _txt_set = titles_set[txt_i[j]]
        _txt_cnt = txt_cnt_all[txt_i[j]]
        _txt_avg = txt_avg_all[txt_i[j]]
        _txt_std = txt_std_all[txt_i[j]]
        diff_txt_set = set(titles[txt_i[j]]) & txt_set
        diff_txt_set = len(numset & diff_txt_set) / (len(diff_txt_set) + 1)
        xor_txt_set = set(titles[txt_i[j]]) ^ txt_set
        xor_txt_set = len(numset & xor_txt_set) / (len(xor_txt_set) + 1)
        jac_txt = len(txt_set & _txt_set) / (len(txt_set | _txt_set) + 1)
        lev_dist = Levenshtein.distance(titles[i], titles[txt_i[j]])
        d = {
            'txt_sim': txt_d[j],
            'txt_avg': txt_avg, 
            'txt_std': txt_std,
            'txt_avg2': _txt_avg,
            'txt_std2': _txt_std,

            'txt_avg_raw': txt_avg_raw_all[i],
            'txt_avg2_raw': txt_avg_raw_all[txt_i[j]],

            'jac_txt': jac_txt,
            'diff_txt_set': diff_txt_set, 
            'xor_txt_set': xor_txt_set,
            'lev_dist': lev_dist,
            'len_txt': len(titles[i]), 
            'len_txt2': len(titles[txt_i[j]]),
            'lang_en': int(langs[i] == 'en'),
            'lang_en2': int(langs[txt_i[j]] == 'en'),
        }
        d.update({f'txt_cnt_{ii}': txt_cnt[ii] for ii in range(txt_cnt.shape[0])})
        d.update({f'txt_cnt2_{ii}': _txt_cnt[ii] for ii in range(_txt_cnt.shape[0])})
        if txt_i[j] in map_feat:
            map_feat[txt_i[j]].update(d)
        else:
            map_feat[txt_i[j]] = d
            
    for j in range(K):
        _bth_cnt = bth_cnt_all[bth_i[j]]
        _bth_avg = bth_avg_all[bth_i[j]]
        _bth_std = bth_std_all[bth_i[j]]
        if bth_i[j] in map_feat:
            d = map_feat[bth_i[j]]
        else:
            d = {}
        d.update({
            'bth_sim': bth_d[j],
            'bth_avg': bth_avg, 
            'bth_std': bth_std,
            'bth_avg2': _bth_avg,
            'bth_std2': _bth_std,

            'bth_avg_raw': bth_avg_raw_all[i],
            'bth_avg2_raw': bth_avg_raw_all[bth_i[j]],
        })
        d.update({f'bth_cnt_{ii}': bth_cnt[ii] for ii in range(bth_cnt.shape[0])})
        d.update({f'bth_cnt2_{ii}': _bth_cnt[ii] for ii in range(_bth_cnt.shape[0])})
        if 'lev_dist' not in d:
            _bth_set = titles_set[bth_i[j]] #set(titles[bth_i[j]])
            diff_bth_set = set(titles[bth_i[j]]) & bth_set
            diff_bth_set = len(numset & diff_bth_set) / (len(diff_bth_set) + 1)
            xor_bth_set = set(titles[bth_i[j]]) ^ bth_set
            xor_bth_set = len(numset & xor_bth_set) / (len(xor_bth_set) + 1)
            jac_bth = len(bth_set & _bth_set) / (len(bth_set | _bth_set) + 1)
            lev_dist = Levenshtein.distance(titles[i], titles[bth_i[j]])
            d.update({
                'jac_txt': jac_bth,
                'diff_txt_set': diff_bth_set, 
                'xor_txt_set': xor_bth_set,
                'lev_dist': lev_dist,
                'len_txt': len(titles[i]), 
                'len_txt2': len(titles[bth_i[j]]),
                'lang_en': int(langs[i] == 'en'),
                'lang_en2': int(langs[bth_i[j]] == 'en'),
            })
        if 'img_width' not in d:    
            _w, _h = width_hight[bth_i[j]]
            diff_width = abs(img_width - _w)
            diff_hight = abs(img_hight - _h)
            d.update({
                'diff_width': diff_width,
                'diff_hight': diff_hight,
                 'img_width': img_width,
                 'img_hight': img_hight,
                 'img_width2': _w,
                 'img_hight2': _h,
                
                     'st_size': st_size[i],
                     'st_size2': st_size[bth_i[j]],
                     'wh_st_size': wh_st_size[i],
                     'wh_st_size2': wh_st_size[bth_i[j]]
                     })
        map_feat[bth_i[j]] = d
            
    for j in range(K):
        _mut_cnt = mut_cnt_all[mut_i[j]]
        _mut_avg = mut_avg_all[mut_i[j]]
        _mut_std = mut_std_all[mut_i[j]]
        if mut_i[j] in map_feat:
            d = map_feat[mut_i[j]]
        else:
            d = {}
        d.update({
            'mut_sim': mut_d[j],
            'mut_avg': mut_avg, 
            'mut_std': mut_std,
            'mut_avg2': _mut_avg,
            'mut_std2': _mut_std,
            'mut_avg_raw': mut_avg_raw_all[i],
            'mut_avg2_raw': mut_avg_raw_all[mut_i[j]],
        })
        d.update({f'mut_cnt_{ii}': mut_cnt[ii] for ii in range(mut_cnt.shape[0])})
        d.update({f'mut_cnt2_{ii}': _mut_cnt[ii] for ii in range(_mut_cnt.shape[0])})
        if 'lev_dist' not in d:
            _mut_set = titles_set[mut_i[j]]#set(titles[mut_i[j]])
            diff_mut_set = set(titles[mut_i[j]]) & mut_set
            diff_mut_set = len(numset & diff_mut_set) / (len(diff_mut_set) + 1)
            xor_mut_set = set(titles[mut_i[j]]) ^ mut_set
            xor_mut_set = len(numset & xor_mut_set) / (len(xor_mut_set) + 1)
            jac_mut = len(mut_set & _mut_set) / (len(mut_set | _mut_set) + 1)
            lev_dist = Levenshtein.distance(titles[i], titles[mut_i[j]])
            d.update({
                'jac_txt': jac_mut,
                'diff_txt_set': diff_mut_set, 
                'xor_txt_set': xor_mut_set,
                'lev_dist': lev_dist,
                'len_txt': len(titles[i]), 
                'len_txt2': len(titles[mut_i[j]]),
                'lang_en': int(langs[i] == 'en'),
                'lang_en2': int(langs[mut_i[j]] == 'en'),
            })
        if 'img_width' not in d:    
            _w, _h = width_hight[mut_i[j]]
            diff_width = abs(img_width - _w)
            diff_hight = abs(img_hight - _h)
            d.update({
                'diff_width': diff_width,
                'diff_hight': diff_hight,
                'img_width': img_width,
                'img_hight': img_hight,
                'img_width2': _w,
                'img_hight2': _h,
                'st_size': st_size[i],
                'st_size2': st_size[mut_i[j]],
                'wh_st_size': wh_st_size[i],
                'wh_st_size2': wh_st_size[mut_i[j]]
            })
        map_feat[mut_i[j]] = d

    for j in range(K):
        _brt_cnt = brt_cnt_all[brt_i[j]]
        _brt_avg = brt_avg_all[brt_i[j]]
        _brt_std = brt_std_all[brt_i[j]]
        if brt_i[j] in map_feat:
            d = map_feat[brt_i[j]]
        else:
            d = {}
        d.update({
            'brt_sim': brt_d[j],
            'brt_avg': brt_avg, 
            'brt_std': brt_std,
            'brt_avg2': _brt_avg,
            'brt_std2': _brt_std,
            'brt_avg_raw': brt_avg_raw_all[i],
            'brt_avg2_raw': brt_avg_raw_all[brt_i[j]],
        })
        d.update({f'brt_cnt_{ii}': brt_cnt[ii] for ii in range(brt_cnt.shape[0])})
        d.update({f'brt_cnt2_{ii}': _brt_cnt[ii] for ii in range(_brt_cnt.shape[0])})
        if 'lev_dist' not in d:
            _brt_set = titles_set[brt_i[j]]
            diff_brt_set = set(titles[brt_i[j]]) & brt_set
            diff_brt_set = len(numset & diff_brt_set) / (len(diff_brt_set) + 1)
            xor_brt_set = set(titles[brt_i[j]]) ^ brt_set
            xor_brt_set = len(numset & xor_brt_set) / (len(xor_brt_set) + 1)
            jac_brt = len(brt_set & _brt_set) / (len(brt_set | _brt_set) + 1)
            lev_dist = Levenshtein.distance(titles[i], titles[brt_i[j]])
            d.update({
                'jac_txt': jac_brt,
                'diff_txt_set': diff_brt_set, 
                'xor_txt_set': xor_brt_set,
                'lev_dist': lev_dist,
                'len_txt': len(titles[i]), 
                'len_txt2': len(titles[brt_i[j]]),
                'lang_en': int(langs[i] == 'en'),
                'lang_en2': int(langs[brt_i[j]] == 'en'),
            })
        map_feat[brt_i[j]] = d

    feat[:] = 0 
    for ii, (k, map_val) in enumerate(map_feat.items()):
        list_idx.append(i)
        list_idx2.append(k)
        for c, v in map_val.items():
            if c not in map_col2id:
                map_col2id[c] = len(map_col2id)
            feat[ii, map_col2id[c]] = v

    all_feat[ptr:ptr + len(map_feat)] = feat[:len(map_feat)]
    ptr += len(map_feat)
    
del img_D, img_I, text_D, text_I, bert_D, bert_I, bth_D, bth_I, mut_D, mut_I
gc.collect()

del list_feats
gc.collect()

map_weights = {sim: all_feat[:ptr, map_col2id[f'{sim}_sim']] for sim in ['img', 'bth', 'mut', 'txt', 'brt']}

del feat
gc.collect()

import networkx as nx


list_idx = np.array(list_idx)
list_idx2 = np.array(list_idx2)

from igraph import Graph
map_sim = {}
for sim in tqdm(['img', 'bth', 'mut', 'txt', 'brt'], desc='graph'):
    weights = map_weights[sim]
    idx = weights > 0
    with timer('add edges'):
        g = Graph()
        g.add_vertices(len(df_test))
        g.add_edges(list(zip(list_idx[idx], list_idx2[idx])), {'weight': weights[idx]})
    with timer('pagerank'):
        map_pr = np.array(g.pagerank(damping=0.85, weights='weight', niter=100, eps=1e-06, directed=False))
    with timer('pagerank get'):
        data1 = map_pr[list_idx]
        data2 = map_pr[list_idx2]
        data1[weights <= 0] = 0
        data2[weights <= 0] = 0
        map_sim[f'{sim}_pagerank'] = data1
        map_sim[f'{sim}_pagerank2'] = data2
    del map_pr, g
    gc.collect()

for c, v in tqdm(map_sim.items()):
    map_col2id[c] = len(map_col2id)
    all_feat[:ptr, map_col2id[c]] = v

import treelite_runtime
from cuml import ForestInference
import treelite
import pickle
import lightgbm as lgb

all_weights = {
    '../input/shopee-metric-resnet50d512-0328-newfold/0508_qe_best_0.345/': 1,
}

s = sum(all_weights.values())
all_weights = {k: v / s for k, v in all_weights.items()}
    
list_clf = []
weights = []
thresholds = [] #[0.358, 0.361, 0.350, 0.336, 0.348, 0.346]
for path in [
    '../input/shopee-metric-resnet50d512-0328-newfold/0508_qe_best_0.345/',
    ]:
    name = os.path.dirname(path).split('/')[-1]
    th = float(name.split('_')[-1])
    if all_weights.get(path, 0) == 0:
        continue
        
    fi = ForestInference()
    fi.load_from_treelite_model(treelite.Model.load(f'{path}/all_data_clf_norm.lgb', model_format='lightgbm'))
    list_clf += [fi]
    thresholds += [th]
    weights += [all_weights[path]]
    
print(weights)
print(thresholds)

col = lgb.Booster(model_file=f'{path}/all_data_clf_norm.lgb').feature_name()

for sf in ['img', 'txt', 'mut', 'bth', 'brt']:
    all_feat[:ptr, map_col2id[f'{sf}_avg']] = all_feat[:ptr, map_col2id[f'{sf}_avg_raw']]
    all_feat[:ptr, map_col2id[f'{sf}_avg2']] = all_feat[:ptr, map_col2id[f'{sf}_avg2_raw']]

CHUNK = 1000000
preds = []
col_idx = [map_col2id[c] for c in col]

for ch in tqdm(range(0, ptr, CHUNK), desc='pred chunk'):
    feat = cp.asarray(all_feat[ch:ch+CHUNK, col_idx]).astype('float32')
    probs = np.vstack([(c.predict(feat).get() - thresholds[ii]) * weights[ii] for ii, c in enumerate(list_clf)])
    preds += probs.sum(axis=0).tolist()
    del feat
    gc.collect()

df_pred = pd.DataFrame(
    dict(
        posting_id=list_idx,
        posting_id_target=list_idx2,
        pred=preds[:ptr]
    )
)

idx = df_test.posting_id.values
df_pred['posting_id'] = [idx[i] for i in df_pred['posting_id'].values]
df_pred['posting_id_target'] = [idx[i] for i in df_pred['posting_id_target'].values]

df_pred.to_pickle('submission_tkm.pkl')
