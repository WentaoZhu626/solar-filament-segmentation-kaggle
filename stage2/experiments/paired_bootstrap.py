
import pickle, sys, numpy as np
D = pickle.load(open(sys.argv[1], 'rb')); res = D['res']; folds = np.asarray(D['folds'])
names = [str(k) for k in res[0]]; T = {str(k): np.array([r[k] for r in res]) for k in res[0]}
pq = lambda S: S[..., 0] / (S[..., 1] + 0.5 * S[..., 2] + 0.5 * S[..., 3])
n = len(res); rng = np.random.default_rng(0); B = 2000
W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).astype(float)
for k in names: 
    b = pq(W @ T[k]); print(f'{k:5s} PQ {pq(T[k].sum(0)):.4f}  90% CI [{np.percentile(b, 5):.4f}, {np.percentile(b, 95):.4f}]')
def paired(a, c):
    d = pq(W @ T[a]) - pq(W @ T[c]); d0 = pq(T[a].sum(0)) - pq(T[c].sum(0))
    imgs = (pq(T[a]) if False else None)
    print(f'delta {a} - {c}: {d0:+.4f}  95% CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]  P(delta<=0) = {(d <= 0).mean():.4f}')
k1 = '0.5'
paired(k1, 'orig'); pass
# month-block bootstrap (folds are month groups -> resample folds' images by month id from filename)
mon = np.array([f[:6] for f in D['fn']]); um = np.unique(mon); idx = {m: np.where(mon == m)[0] for m in um}
Wm = np.zeros((B, n))
for b in range(B):
    for m in rng.choice(um, len(um)): Wm[b, idx[m]] += 1
d = pq(Wm @ T[k1]) - pq(Wm @ T['orig'])
print(f'month-block bootstrap ({len(um)} months): delta {k1}-orig 95% CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}], P<=0 {(d <= 0).mean():.4f}')
# public-LB-like subsets: 90 random images
for k in ['orig', k1]:
    sub = np.array([pq(T[k][rng.choice(n, 90, replace=False)].sum(0)) for _ in range(4000)])
    print(f'90-image subsets ({k}): mean {sub.mean():.4f} sd {sub.std():.4f}  5-95% [{np.percentile(sub, 5):.4f}, {np.percentile(sub, 95):.4f}]')
dd = []
for _ in range(4000):
    s = rng.choice(n, 90, replace=False); dd.append(pq(T[k1][s].sum(0)) - pq(T['orig'][s].sum(0)))
dd = np.array(dd); print(f'90-image subsets, delta {k1}-orig: mean {dd.mean():+.4f} sd {dd.std():.4f}; P(subset shows delta<0.005 i.e. invisible at 2 decimals-ish) = {(dd < 0.005).mean():.3f}; P(delta<0) = {(dd < 0).mean():.3f}')
per_img = pq(T[k1]) - pq(T['orig']); ok = np.isfinite(per_img)
print(f'per image: better {(per_img[ok] > 1e-9).sum()}, worse {(per_img[ok] < -1e-9).sum()}, equal {(np.abs(per_img[ok]) <= 1e-9).sum()}')
