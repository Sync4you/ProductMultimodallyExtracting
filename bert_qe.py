import gc
import numpy as np
import faiss

def query_expansion(feats, sims, topk_idx, alpha=0.5, k=2):
    weights = np.expand_dims(sims[:, :k] ** alpha, axis=-1).astype(np.float32)
    feats = (feats[topk_idx[:, :k]] * weights).sum(axis=1)
    return feats

brt_feats = np.load('/tmp/bert_feats.npy')

res = faiss.StandardGpuResources()
index_brt = faiss.IndexFlatIP(brt_feats.shape[1])
index_brt = faiss.index_cpu_to_gpu(res, 0, index_brt)
index_brt.add(brt_feats)
brt_D, brt_I = index_brt.search(brt_feats, 60)

np.save('/tmp/brt_D', brt_D)
np.save('/tmp/brt_I', brt_I)

del index_brt
gc.collect()

brt_feats_qe = query_expansion(brt_feats, brt_D, brt_I)
brt_feats_qe /= np.linalg.norm(brt_feats_qe, 2, axis=1, keepdims=True)

brt_feats = np.hstack([brt_feats, brt_feats_qe])
brt_feats /= np.linalg.norm(brt_feats, axis=1).reshape((-1, 1))

index = faiss.IndexFlatIP(brt_feats.shape[1])
res = faiss.StandardGpuResources()
index = faiss.index_cpu_to_gpu(res, 0, index)

index.add(brt_feats)
brt_D, brt_I = index.search(brt_feats, 60)

np.save('/tmp/brt_D_qe', brt_D)
np.save('/tmp/brt_I_qe', brt_I)
print('end')
