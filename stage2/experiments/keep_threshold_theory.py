
import pickle, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
V = pickle.load(open('/root/autodl-tmp/runs/vis/vis.pkl', 'rb')); REC = V['rec']
CO = '/root/autodl-tmp/runs/cnnf/'
cn = np.mean([np.load(CO + f'oof{t}.npy') for t in ['', '_cnxt', '_effb3', '_r50', '_r34_192']], 0)
own = np.concatenate([np.full(len(r['kept']), n) for n, r in enumerate(REC)])
fold = np.array([REC[n]['fold'] for n in own])
X = np.concatenate([r['X'] for r in REC]); gb = np.concatenate([r['score'] for r in REC])
IOU = [r['iou'] for r in REC]
y = np.concatenate([(r['iou'] > 0.5).mean(1) for r in REC])                       # share of views matched
u = np.concatenate([np.where(r['iou'] > 0.5, r['iou'], 0).mean(1) for r in REC])   # expected matched IoU
m = np.concatenate([r['iou'][r['iou'] > 0.5] for r in REC])
print(f'candidates {len(y)} | matched pairs {len(m)}: E[IoU|match] = {m.mean():.4f} (median {np.median(m):.3f}, sd {m.std():.3f})')
q_c = np.where(y > 0, u / np.maximum(y, 1e-9), np.nan)
print(f'per-candidate q = E[IoU|match]: sd across candidates {np.nanstd(q_c):.3f}; corr(y, u) = {np.corrcoef(y, u)[0, 1]:.4f}')
ngt = [np.array([int(r['view_S'][v][1] + r['view_S'][v][3]) for v in range(r['n_views'])]) for r in REC]
def pq(keep_all):
    S = np.zeros(4); k0 = 0
    for n, r in enumerate(REC):
        nc = len(r['kept']); kp = keep_all[k0:k0 + nc]; k0 += nc
        for v in range(r['n_views']):
            best = {}
            for c in np.where(kp & (r['iou'][:, v] > 0.5))[0]:
                g = r['gid'][c, v]
                if r['iou'][c, v] > best.get(g, (0, -1))[0]: best[g] = (r['iou'][c, v], c)
            tp = len(best); S += [sum(x[0] for x in best.values()), tp, kp.sum() - tp, ngt[n][v] - tp]
    return S[0] / (S[1] + 0.5 * S[2] + 0.5 * S[3]), S
print('baseline (vis kept)', round(pq(np.concatenate([r['kept'] for r in REC]))[0], 4))
# oracle: keep iff true u > tau ; theory says tau* = PQ*/2
for tau in [0.15, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.35]:
    print(f'oracle keep u > {tau}: PQ {pq(u > tau)[0]:.4f}')
# learned: OOF GBM predicting y vs predicting u from the 13 features + CNN mean
F = np.column_stack([X, cn])
gbm = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
py, pu = np.zeros(len(y)), np.zeros(len(y))
for k in range(5):
    tr, te = fold != k, fold == k
    py[te] = gbm().fit(F[tr], y[tr]).predict(F[te]); pu[te] = gbm().fit(F[tr], u[tr]).predict(F[te])
qbar = m.mean()
res = {}
for name, sc, taus in [('y-hat', py, np.arange(0.25, 0.5, 0.025)), ('u-hat', pu, np.arange(0.15, 0.36, 0.02)), ('y-hat x qbar', py * qbar, np.arange(0.15, 0.36, 0.02))]:
    out = [(round(t, 3), round(pq(sc >= t)[0], 4)) for t in taus]
    b = max(out, key=lambda x: x[1]); res[name] = b
    print(name, out, '-> best', b)
# self-consistent tau: iterate tau = PQ(tau)/2 (Dinkelbach) with u-hat
tau = 0.2
for it in range(6):
    p_, _ = pq(pu >= tau); print(f'Dinkelbach it {it}: tau {tau:.4f} -> PQ {p_:.4f}'); tau = p_ / 2
