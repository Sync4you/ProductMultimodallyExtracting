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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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

class GraphDataset(Dataset):

    def __init__(self, feats=None, labels=None, weights=None, pair_tuples=None, k=50, top_neighbors=None):
        self.feats = feats
        self.labels = labels
        self.weights = weights
        self.pair_tuples = pair_tuples
        self.k = k
        self.top_neighbors = top_neighbors

    def __getitem__(self, index):
        i, j = self.pair_tuples[index]
        feat = torch.FloatTensor(self.feats[i][j])

        padding_i = [[0] * feat.shape[0]] * (self.k - len(self.top_neighbors[i]))
        neighbor_feats_i = torch.FloatTensor([
            self.feats[i][neighbor]
            for neighbor in self.top_neighbors[i]
        ] + padding_i)
        padding_j = [[0] * feat.shape[0]] * (self.k - len(self.top_neighbors[j]))
        neighbor_feats_j = torch.FloatTensor([
            self.feats[j][neighbor]
            for neighbor in self.top_neighbors[j]
        ] + padding_j)
        neighbor_feats = torch.cat([feat.unsqueeze(0), neighbor_feats_i, neighbor_feats_j], dim=0)

        outputs = (feat, neighbor_feats)
        if self.labels is not None:
            outputs += (self.labels[i] == self.labels[j],)
        if self.weights is not None:
            outputs += (self.weights[i],)

        return outputs

    def __len__(self):
        return len(self.pair_tuples)


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h):
        Wh = h @ self.W  # h.shape: (B, N, in_features), Wh.shape: (B, N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        attention = F.softmax(e, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        B, N, D = Wh.shape

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        return all_combinations_matrix.view(-1, N, N, 2 * D)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GATPairClassifier(nn.Module):
    def __init__(self, nfeat, nhid=8, nclass=1, dropout=0.6, alpha=0.2, nheads=8, pooling='first'):
        super().__init__()
        self.dropout = dropout
        self.pooling = pooling

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)

        self.classifier = nn.Sequential(
            nn.Linear(nfeat + nhid, nhid),
            nn.PReLU(),
            nn.BatchNorm1d(nhid),
            nn.Linear(nhid, nclass),
        )

    def forward_gat(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x))
        if self.pooling == 'first':
            return x[:, 0]
        elif self.pooling == 'mean':
            return x.mean(dim=1)

    def forward(self, feats, neighbor_feats):
        gat_feats = self.forward_gat(neighbor_feats)
        cat_feats = torch.cat([feats, gat_feats], dim=1)
        return self.classifier(cat_feats).squeeze(1)


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
simmat_tfidf = tfidf_feats @ tfidf_feats.T

with timer('load'):
    st_sizes, img_hs, img_ws = joblib.load('/tmp/lyk_img_meta_data.pkl')
    similarities_img = np.load('/tmp/img_D_qe.npy')[:, :k]
    indexes_img = np.load('/tmp/img_I_qe.npy')[:, :k]

    similarities_bert = np.load('/tmp/brt_D_qe.npy')[:, :k]
    indexes_bert = np.load('/tmp/brt_I_qe.npy')[:, :k]

    similarities_mm = np.load('/tmp/mut_D_qe.npy')[:, :k]
    indexes_mm = np.load('/tmp/mut_I_qe.npy')[:, :k]
    
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

ckpt = torch.load('../input/shopee-meta-models/v135.pth')
params = ckpt['params']

top_neighbors = defaultdict(list)
feats = defaultdict(lambda: defaultdict())

pair_tuples = []
for i in tqdm(range(len(df))):
    right_indexes = set(indexes_img[i, :k].tolist() + indexes_bert[i, :k].tolist())
    right_indexes.remove(i)  # remove self

    right_indexes = list(right_indexes)
    scores = {}
    for j in right_indexes:
        pair_tuples.append((i, j))

        sim_img = simmat_img.get((i, j), 0)
        sim_bert = simmat_bert.get((i, j), 0)
        sim_mm = simmat_mm.get((i, j), 0)
        sim_tfidf = simmat_tfidf[i, j]
        if sim_img == 0 and sim_bert == 0:
            continue

        feats[i][j] = [
            sim_img,
            sim_tfidf,
            sim_bert,
            sim_mm,
        ]
        scores[j] = sim_img + sim_tfidf + sim_bert + sim_mm

    top_neighbors[i] = sorted(right_indexes, key=lambda x: scores[x], reverse=True)[:params['k']]

dataset = GraphDataset(
    feats=feats,
    pair_tuples=pair_tuples,
    k=params['k'],
    top_neighbors=top_neighbors,
)
loader = DataLoader(dataset, batch_size=2 ** 12, shuffle=False, drop_last=False, num_workers=2, pin_memory=True)

gat = GATPairClassifier(nfeat=len(feats[i][j]), nhid=params['nhid'],
                        dropout=params['dropout'], nheads=params['nheads'], pooling=params['pooling'])
gat.to('cuda').eval()
gat.load_state_dict(ckpt['model'])

del tfidf_feats
gc.collect()
###

preds = []
for feats, neighbor_feats in tqdm(loader, desc='predict', leave=False):
    feats = feats.to('cuda', non_blocking=True)
    neighbor_feats = neighbor_feats.to('cuda', non_blocking=True)
    with torch.no_grad():
        pred = gat(feats, neighbor_feats).sigmoid().detach().cpu().numpy().tolist()
        preds.extend(pred)

conf_th_gcn = 0.3
df_pair = pd.DataFrame()
col, row = list(zip(*pair_tuples))
df_pair['i'] = col
df_pair['j'] = row

df_pair['posting_id'] = df['posting_id'].values[df_pair['i'].values]
df_pair['posting_id_target'] = df['posting_id'].values[df_pair['j'].values]

df_pair = df_pair[['posting_id', 'posting_id_target']]
df_pair['pred'] = preds
df_pair['pred'] -= conf_th_gcn

df_pair.to_pickle('submission_lyak_gcn.pkl')
