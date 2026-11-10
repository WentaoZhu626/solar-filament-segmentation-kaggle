
# Uncertainty features (their idea): disagreement of the 3 ensemble members inside every candidate, added to the GBM filter.
import pickle, numpy as np, cv2
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
V = pickle.load(open('/root/autodl-tmp/runs/vis/vis.pkl', 'rb')); REC = V['rec']
PATHS = [f'/root/autodl-tmp/runs/m_{m}/oof_maps.npy' for m in ['cnxt', 'effb4', 'regy']]
def feats(i, B_png, F_png, cand_src):
    dec = lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
    labs = {'B': dec(B_png), 0.3: dec(F_png)}
    M = np.stack([np.load(p, mmap_mode='r')[i][0].astype(np.float32) / 255 for p in PATHS])
    sd = M.std(0); rng = M.max(0) - M.min(0)
    out = []
    for s_, lid in cand_src:
        m = labs[s_] == lid
        if not m.any(): out.append(np.zeros(7, np.float32)); continue
        pm = M[:, m].mean(1)
        out.append(np.array([sd[m].mean(), np.percentile(sd[m], 90), rng[m].mean(), pm.min(), pm.max() - pm.min(),
                             ((M[:, m] > 0.5).mean(1) > 0.5).sum() / 3.0, (sd[m] > 0.15).mean()], np.float32))
    return np.array(out, np.float32).reshape(-1, 7)
U = Parallel(n_jobs=14, batch_size=2)(delayed(feats)(r['i'], r['B_png'], r['F_png'], r['cand_src']) for r in REC)
Uall = np.concatenate(U); np.save('/root/cg/unc_feats.npy', Uall); print('features', Uall.shape, flush=True)
X = np.concatenate([r['X'] for r in REC]); fold = np.concatenate([np.full(len(r['kept']), r['fold']) for r in REC])
y = np.concatenate([(r['iou'] > 0.5).mean(1) for r in REC])
CO = '/root/autodl-tmp/runs/cnnf/'; cn = np.mean([np.load(CO + f'oof{t}.npy') for t in ['', '_cnxt', '_effb3', '_r50', '_r34_192']], 0)
ngt = [np.array([int(r['view_S'][v][1] + r['view_S'][v][3]) for v in range(r['n_views'])]) for r in REC]
def pq(keep):
    S = np.zeros(4); k0 = 0
    for n, r in enumerate(REC):
        nc = len(r['kept']); kp = keep[k0:k0 + nc]; k0 += nc
        for v in range(r['n_views']):
            best = {}
            for c in np.where(kp & (r['iou'][:, v] > 0.5))[0]:
                g = r['gid'][c, v]; best[g] = max(best.get(g, 0), r['iou'][c, v])
            tp = len(best); S += [sum(best.values()), tp, kp.sum() - tp, ngt[n][v] - tp]
    return S[0] / (S[1] + 0.5 * S[2] + 0.5 * S[3])
def oof(F, seeds=3):
    sc = np.zeros(len(y))
    for k in range(5):
        tr, te = fold != k, fold == k
        sc[te] = np.mean([HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=s).fit(F[tr], y[tr]).predict(F[te]) for s in range(seeds)], 0)
    return sc
g0 = oof(X); g1 = oof(np.hstack([X, Uall]))
def best(sc, ths=np.arange(0.3, 0.5, 0.025)):
    r = {round(float(t), 3): round(pq(sc >= t), 4) for t in ths}; b = max(r, key=r.get); return b, r[b], r
for name, sc in [('GBM 13 feats', g0), ('GBM 13 + uncertainty', g1), ('blend CNN 0.6 + GBM13 0.4', 0.6 * cn + 0.4 * g0), ('blend CNN 0.6 + GBM(13+unc) 0.4', 0.6 * cn + 0.4 * g1)]:
    b, v, r = best(sc); print(f'{name:36s} best thr {b} PQ {v}  sweep {r}', flush=True)
from sklearn.metrics import roc_auc_score
print('AUC(y>0.5):', {n: round(roc_auc_score(y > 0.5, s), 4) for n, s in [('g0', g0), ('g1', g1), ('cnn', cn), ('blend0', 0.6 * cn + 0.4 * g0), ('blend1', 0.6 * cn + 0.4 * g1)]})
print('corr(unc_sd_mean, y) =', round(np.corrcoef(Uall[:, 0], y)[0, 1], 3), '| corr(spread member means, y) =', round(np.corrcoef(Uall[:, 4], y)[0, 1], 3))
print('UNC DONE')
