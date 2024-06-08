import gc
import numpy as np
import faiss

def query_expansion(feats, sims, topk_idx, alpha=0.5, k=2):
    weights = np.expand_dims(sims[:, :k] ** alpha, axis=-1).astype(np.float32)
    feats = (feats[topk_idx[:, :k]] * weights).sum(axis=1)
    return feats

mm_feats = np.load('/tmp/mm_feats.npy')

res = faiss.StandardGpuResources()
index_mm = faiss.IndexFlatIP(mm_feats.shape[1])
index_mm = faiss.index_cpu_to_gpu(res, 0, index_mm)
index_mm.add(mm_feats)
mm_D, mm_I = index_mm.search(mm_feats, 60)

np.save('/tmp/mut_D', mm_D)
np.save('/tmp/mut_I', mm_I)

mm_feats_qe = query_expansion(mm_feats, mm_D, mm_I)
mm_feats_qe /= np.linalg.norm(mm_feats_qe, 2, axis=1, keepdims=True)

mm_feats = np.hstack([mm_feats, mm_feats_qe])
mm_feats /= np.linalg.norm(mm_feats, axis=1).reshape((-1, 1))

index = faiss.IndexFlatIP(mm_feats.shape[1])
res = faiss.StandardGpuResources()
index = faiss.index_cpu_to_gpu(res, 0, index)

index.add(mm_feats)
mm_D, mm_I = index.search(mm_feats, 60)

np.save('/tmp/mut_D_qe', mm_D)
np.save('/tmp/mut_I_qe', mm_I)
