import gc
import numpy as np
import faiss

def query_expansion(feats, sims, topk_idx, alpha=0.5, k=2):
    weights = np.expand_dims(sims[:, :k] ** alpha, axis=-1).astype(np.float32)
    feats = (feats[topk_idx[:, :k]] * weights).sum(axis=1)
    return feats

img_feats = np.load('/tmp/img_feats.npy')

res = faiss.StandardGpuResources()
index_img = faiss.IndexFlatIP(img_feats.shape[1])
index_img = faiss.index_cpu_to_gpu(res, 0, index_img)
index_img.add(img_feats)
img_D, img_I = index_img.search(img_feats, 60)

np.save('/tmp/img_D', img_D)
np.save('/tmp/img_I', img_I)

img_feats_qe = query_expansion(img_feats, img_D, img_I)
img_feats_qe /= np.linalg.norm(img_feats_qe, 2, axis=1, keepdims=True)

img_feats = np.hstack([img_feats, img_feats_qe])
img_feats /= np.linalg.norm(img_feats, axis=1).reshape((-1, 1))

index = faiss.IndexFlatIP(img_feats.shape[1])
res = faiss.StandardGpuResources()
index = faiss.index_cpu_to_gpu(res, 0, index)

index.add(img_feats)
img_D, img_I = index.search(img_feats, 60)

np.save('/tmp/img_D_qe', img_D)
np.save('/tmp/img_I_qe', img_I)

print('end')
