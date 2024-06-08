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

    
class ShopeeNet(nn.Module):

    def __init__(self,
                 backbone,
                 num_classes,
                 fc_dim=512,
                 s=30, margin=0.5, p=3):
        super(ShopeeNet, self).__init__()

        self.backbone = backbone
        self.backbone.reset_classifier(num_classes=0)  # remove classifier

        self.fc = nn.Linear(self.backbone.num_features, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        self.p = p

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone.forward_features(x)
        if isinstance(x, tuple):
            x = (x[0] + x[1]) / 2
            x = self.bn(x)
        else:
            x = gem(x, p=self.p).view(batch_size, -1)
            x = self.fc(x)
            x = self.bn(x)
        return x

    def forward(self, x, label):
        feat = self.extract_feat(x)
        x = self.loss_module(feat, label)
        return x, feat


class ShopeeDataset(Dataset):

    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = read_image(str(self.img_dir / row['image_name']))
        _, h, w = img.shape
        st_size = (self.img_dir / row['image_name']).stat().st_size
        if self.transform is not None:
            img = self.transform(img)

        return img, row['title'], h, w, st_size

    def __len__(self):
        return len(self.df)


class MultiModalNet(nn.Module):

    def __init__(self,
                 backbone,
                 bert_model,
                 num_classes,
                 tokenizer,
                 max_len=32,
                 fc_dim=512,
                 s=30, margin=0.5, p=3, loss='ArcMarginProduct'):
        super().__init__()

        self.backbone = backbone
        self.backbone.reset_classifier(num_classes=0)  # remove classifier

        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.fc = nn.Linear(self.bert_model.config.hidden_size + self.backbone.num_features, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        self.p = p

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def extract_feat(self, img, title):
        batch_size = img.shape[0]
        img = self.backbone.forward_features(img)
        img = gem(img, p=self.p).view(batch_size, -1)

        tokenizer_output = self.tokenizer(title, truncation=True, padding=True, max_length=self.max_len)
        input_ids = torch.LongTensor(tokenizer_output['input_ids']).to('cuda')
        token_type_ids = torch.LongTensor(tokenizer_output['token_type_ids']).to('cuda')
        attention_mask = torch.LongTensor(tokenizer_output['attention_mask']).to('cuda')
        title = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # x = x.last_hidden_state.sum(dim=1) / attention_mask.sum(dim=1, keepdims=True)
        title = title.last_hidden_state.mean(dim=1)

        x = torch.cat([img, title], dim=1)
        x = self.fc(x)
        x = self.bn(x)
        return x


####

df, img_dir = load_data()
    
###

checkpoint1 = torch.load('../input/shopee/v45.pth')
checkpoint2 = torch.load('../input/shopee/v34.pth')
checkpoint3 = torch.load('../input/shopee/v79.pth')
params1 = checkpoint1['params']
params2 = checkpoint2['params']
params3 = checkpoint3['params']

transform = Compose([
    Resize(size=params1['test_size'] + 32, interpolation=Image.BICUBIC),
    CenterCrop((params1['test_size'], params1['test_size'])),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = ShopeeDataset(df=df, img_dir=img_dir, transform=None)
data_loader = DataLoader(dataset, batch_size=8, shuffle=False,
                         drop_last=False, pin_memory=True, num_workers=NUM_WORKERS, collate_fn=lambda x: x)

backbone = timm.create_model(model_name=params1['backbone'], pretrained=False)
model1 = ShopeeNet(backbone, num_classes=0, fc_dim=params1['fc_dim'])
model1 = model1.to('cuda')
model1.load_state_dict(checkpoint1['model'], strict=False)
model1.train(False)
model1.p = params1['p_eval']

backbone = timm.create_model(model_name=params2['backbone'], pretrained=False)
model2 = ShopeeNet(backbone, num_classes=0, fc_dim=params2['fc_dim'])
model2 = model2.to('cuda')
model2.load_state_dict(checkpoint2['model'], strict=False)
model2.train(False)
model2.p = params2['p_eval']

backbone = timm.create_model(model_name=params3['backbone'], pretrained=False)
tokenizer = BertTokenizerFast(vocab_file='../input/bert-indo/vocab.txt')
bert_config = BertConfig.from_json_file('../input/bert-indo/config.json')
bert_model = BertModel(bert_config)
model3 = MultiModalNet(backbone, bert_model, num_classes=0, tokenizer=tokenizer, max_len=params3['max_len'],
                       fc_dim=params3['fc_dim'], s=params3['s'], margin=params3['margin'], loss=params3['loss'])
model3 = model3.to('cuda')
model3.load_state_dict(checkpoint3['model'], strict=False)
model3.train(False)
model3.p = params3['p_eval']

img_feats1 = []
img_feats2 = []
mm_feats = []
img_hs = []
img_ws = []
st_sizes = []
for batch in tqdm(data_loader, total=len(data_loader), miniters=None, ncols=55):
    img, title, h, w, st_size = list(zip(*batch))
    img = torch.cat([transform(x.to('cuda').float() / 255)[None] for x in img], axis=0)
    title = list(title)
    with torch.no_grad():
        feats_minibatch1 = model1.extract_feat(img)
        img_feats1.append(feats_minibatch1.cpu().numpy())
        feats_minibatch2 = model2.extract_feat(img)
        img_feats2.append(feats_minibatch2.cpu().numpy())
        feats_minibatch3 = model3.extract_feat(img, title)
        mm_feats.append(feats_minibatch3.cpu().numpy())
    img_hs.extend(list(h))
    img_ws.extend(list(w))
    st_sizes.extend(list(st_size))

img_feats1 = np.concatenate(img_feats1)
img_feats1 /= np.linalg.norm(img_feats1, 2, axis=1, keepdims=True)
img_feats2 = np.concatenate(img_feats2)
img_feats2 /= np.linalg.norm(img_feats2, 2, axis=1, keepdims=True)
mm_feats = np.concatenate(mm_feats)
mm_feats /= np.linalg.norm(mm_feats, 2, axis=1, keepdims=True)

np.save('/tmp/img_feats1', img_feats1)
np.save('/tmp/img_feats2', img_feats2)

img_feats = np.concatenate([
    img_feats1 * 1.0,
    img_feats2 * 1.0,
], axis=1)
img_feats /= np.linalg.norm(img_feats, 2, axis=1, keepdims=True)
###

np.save('/tmp/img_feats', img_feats)

res = faiss.StandardGpuResources()
index_img = faiss.IndexFlatIP(params1['fc_dim'] + params2['fc_dim'])
index_img = faiss.index_cpu_to_gpu(res, 0, index_img)
index_img.add(img_feats)
similarities_img, indexes_img = index_img.search(img_feats, k)


joblib.dump([similarities_img, indexes_img], '/tmp/lyk_img_data.pkl')
joblib.dump([st_sizes, img_hs, img_ws], '/tmp/lyk_img_meta_data.pkl')

res = faiss.StandardGpuResources()
index_mm = faiss.IndexFlatIP(params3['fc_dim'])
index_mm = faiss.index_cpu_to_gpu(res, 0, index_mm)
index_mm.add(mm_feats)
similarities_mm, indexes_mm = index_mm.search(mm_feats, k)

joblib.dump([similarities_mm, indexes_mm], '/tmp/lyk_mm_data.pkl')

### for TKM
np.save('/tmp/mm_feats', mm_feats)
