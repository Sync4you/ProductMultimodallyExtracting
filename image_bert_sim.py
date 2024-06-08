import gc
import numpy as np
import faiss

def query_expansion(feats, sims, topk_idx, alpha=0.5, k=2):
    weights = np.expand_dims(sims[:, :k] ** alpha, axis=-1).astype(np.float32)
    feats = (feats[topk_idx[:, :k]] * weights).sum(axis=1)
    return feats


feats_bert = np.load('/tmp/bert_feats.npy')
feats_img = np.load('/tmp/img_feats.npy')

bth_feats = np.hstack([feats_bert, feats_img])
bth_feats /= np.linalg.norm(bth_feats, 2, axis=1, keepdims=True)

print(bth_feats.shape)

res = faiss.StandardGpuResources()
index = faiss.IndexFlatIP(bth_feats.shape[1])
index = faiss.index_cpu_to_gpu(res, 0, index)
index.add(bth_feats)

bth_D, bth_I = index.search(bth_feats, 60)
np.save('/tmp/bth_D', bth_D)
np.save('/tmp/bth_I', bth_I)

del index
gc.collect()

bth_feats_qe = query_expansion(bth_feats, bth_D, bth_I)
bth_feats_qe /= np.linalg.norm(bth_feats_qe, 2, axis=1, keepdims=True)

bth_feats = np.hstack([bth_feats, bth_feats_qe])
bth_feats /= np.linalg.norm(bth_feats, axis=1).reshape((-1, 1))

index = faiss.IndexFlatIP(bth_feats.shape[1])
res = faiss.StandardGpuResources()
index = faiss.index_cpu_to_gpu(res, 0, index)

index.add(bth_feats)
bth_D, bth_I = index.search(bth_feats, 60)

np.save('/tmp/bth_D_qe', bth_D)
np.save('/tmp/bth_I_qe', bth_I)
print('end')
