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
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import Resize, RandomHorizontalFlip, ColorJitter, Normalize, Compose, RandomResizedCrop, CenterCrop, ToTensor

from tqdm import tqdm
from PIL import Image
import joblib
from scipy.sparse import hstack, vstack, csc_matrix, csr_matrix
import editdistance
import networkx as nx

from transformers import BertConfig, BertModel, BertTokenizerFast

NUM_CLASSES = 11014
NUM_WORKERS = 2
SEED = 0


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class BertNet(nn.Module):

    def __init__(self,
                 bert_model,
                 num_classes,
                 tokenizer,
                 max_len=32,
                 fc_dim=512,
                 simple_mean=True,
                 s=30, margin=0.5, p=3, loss='ArcMarginProduct'):
        super().__init__()

        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.fc = nn.Linear(self.bert_model.config.hidden_size, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        self.p = p
        self.simple_mean = simple_mean

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def extract_feat(self, x):
        tokenizer_output = self.tokenizer(x, truncation=True, padding=True, max_length=self.max_len)
        if 'token_type_ids' in tokenizer_output:
            input_ids = torch.LongTensor(tokenizer_output['input_ids']).to('cuda')
            token_type_ids = torch.LongTensor(tokenizer_output['token_type_ids']).to('cuda')
            attention_mask = torch.LongTensor(tokenizer_output['attention_mask']).to('cuda')
            x = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        else:
            input_ids = torch.LongTensor(tokenizer_output['input_ids']).to('cuda')
            attention_mask = torch.LongTensor(tokenizer_output['attention_mask']).to('cuda')
            x = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        if self.simple_mean:
            x = x.last_hidden_state.mean(dim=1)
        else:
            x = torch.sum(x.last_hidden_state * attention_mask.unsqueeze(-1), dim=1) / attention_mask.sum(dim=1, keepdims=True)
        x = self.fc(x)
        x = self.bn(x)
        return x


class BertDataset(Dataset):

    def __init__(self, df):
        self.df = df

    def __getitem__(self, index):
        row = self.df.iloc[index]

        if 'y' in row.keys():
            target = torch.tensor(row['y'], dtype=torch.long)
            return row['title'], target
        else:
            return row['title']

    def __len__(self):
        return len(self.df)

df, img_dir = load_data()

checkpoint = torch.load('../input/shopee/v75.pth')
checkpoint2 = torch.load('../input/shopee/v102.pth')
checkpoint3 = torch.load('../input/shopee/v103.pth')

params_bert = checkpoint['params']
params_bert2 = checkpoint2['params']
params_bert3 = checkpoint3['params']

datasets = {
    'valid': BertDataset(df=df)
}
data_loaders = {
    'valid': DataLoader(datasets['valid'], batch_size=params_bert['batch_size'] * 2, shuffle=False,
                        drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
}

tokenizer = BertTokenizerFast(vocab_file='../input/bert-indo/vocab.txt')
bert_config = BertConfig.from_json_file('../input/bert-indo/config.json')
bert_model = BertModel(bert_config)
model = BertNet(bert_model, num_classes=0, tokenizer=tokenizer, max_len=params_bert['max_len'], simple_mean=True,
                fc_dim=params_bert['fc_dim'], s=params_bert['s'], margin=params_bert['margin'], loss=params_bert['loss'])

model = model.to('cuda')
model.load_state_dict(checkpoint['model'], strict=False)
model.train(False)

from transformers import AutoTokenizer, AutoModel, AutoConfig

model_name = params_bert2['model_name']
tokenizer = AutoTokenizer.from_pretrained('../input/bertmultilingual/')
bert_config = AutoConfig.from_pretrained('../input/bertmultilingual/')
bert_model = AutoModel.from_config(bert_config)
model2 = BertNet(bert_model, num_classes=0, tokenizer=tokenizer, max_len=params_bert['max_len'], simple_mean=False,
                 fc_dim=params_bert['fc_dim'], s=params_bert['s'], margin=params_bert['margin'], loss=params_bert['loss'])
model2 = model2.to('cuda')
model2.load_state_dict(checkpoint2['model'], strict=False)
model2.train(False)

#########

model_name = params_bert3['model_name']
tokenizer = AutoTokenizer.from_pretrained('../input/bertxlm/')
bert_config = AutoConfig.from_pretrained('../input/bertxlm/')
bert_model = AutoModel.from_config(bert_config)
model3 = BertNet(bert_model, num_classes=0, tokenizer=tokenizer, max_len=params_bert3['max_len'], simple_mean=False,
                 fc_dim=params_bert3['fc_dim'], s=params_bert3['s'], margin=params_bert3['margin'], loss=params_bert3['loss'])
model3 = model3.to('cuda')
model3.load_state_dict(checkpoint3['model'], strict=False)
model3.train(False)

bert_feats1 = []
bert_feats2 = []
bert_feats3 = []
for i, title in tqdm(enumerate(data_loaders['valid']),
                     total=len(data_loaders['valid']), miniters=None, ncols=55):
    with torch.no_grad():
        bert_feats_minibatch = model.extract_feat(title)
        bert_feats1.append(bert_feats_minibatch.cpu().numpy())
        bert_feats_minibatch = model2.extract_feat(title)
        bert_feats2.append(bert_feats_minibatch.cpu().numpy())
        bert_feats_minibatch = model3.extract_feat(title)
        bert_feats3.append(bert_feats_minibatch.cpu().numpy())

bert_feats1 = np.concatenate(bert_feats1)
bert_feats1 /= np.linalg.norm(bert_feats1, 2, axis=1, keepdims=True)
bert_feats2 = np.concatenate(bert_feats2)
bert_feats2 /= np.linalg.norm(bert_feats2, 2, axis=1, keepdims=True)
bert_feats3 = np.concatenate(bert_feats3)
bert_feats3 /= np.linalg.norm(bert_feats3, 2, axis=1, keepdims=True)

bert_feats = np.concatenate([bert_feats1, bert_feats2], axis=1)
bert_feats /= np.linalg.norm(bert_feats, 2, axis=1, keepdims=True)

res = faiss.StandardGpuResources()
index_bert = faiss.IndexFlatIP(params_bert['fc_dim'])
index_bert = faiss.index_cpu_to_gpu(res, 0, index_bert)
index_bert.add(bert_feats1)
similarities_bert, indexes_bert = index_bert.search(bert_feats1, k)

np.save('/tmp/bert_feats1', bert_feats1)
np.save('/tmp/bert_feats2', bert_feats2)
np.save('/tmp/bert_feats3', bert_feats3)

bert_feats = np.concatenate([bert_feats1, bert_feats2, bert_feats3], axis=1)
bert_feats /= np.linalg.norm(bert_feats, 2, axis=1, keepdims=True)

np.save('/tmp/bert_feats', bert_feats)

joblib.dump([similarities_bert, indexes_bert], '/tmp/lyk_bert_data.pkl')
