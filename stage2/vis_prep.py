# Rebuild the final ens3 OOF instances (B + faint p>0.3 + out-of-fold FP filter) from the per-model OOF maps and store,
# for every training observation, the candidate label maps + per-candidate score / kept flag / IoU with each annotator.
# Output: /root/autodl-tmp/runs/vis/vis.pkl  (small; read by the error-browser notebook, CPU only)
import sys, itertools, os, pickle
sys.argv = ['x', '--tag', 'err3', '--eval_only']
src = open('/root/autodl-tmp/run_ens.py').read()
def chunk(a, b):
    i, j = src.index(a), src.index(b); assert i < j; return src[i:j]
exec(src[:src.index('# ---------- 3) 5 folds')])
exec(chunk('# ---------- 3) 5 folds', '# ---------- 4) training samples'))
exec(chunk('# ---------- 7) evaluation', 'ALL = np.arange(len(OBS))'))
exec(chunk('# ===================== 8) post-processing B', '# ---------- 9) B on all OOF'))
exec(chunk('from scipy import ndimage as ndi', '# ---- 0) why are'))
ALL = np.arange(len(OBS))
exec(chunk('# ---- 1) configurations', 'CFGS = ['))
from sklearn.ensemble import HistGradientBoostingRegressor
MEMBERS = ['cnxt', 'effb4', 'regy']
PATHS = {m: f'/root/autodl-tmp/runs/m_{m}/oof_maps.npy' for m in MEMBERS}
NJ = int(os.environ.get('NJ', 48))
t0 = time.time()
def _gts(vs): return [gt_view(v) for v in vs]
GT = dict(zip(ALL, Parallel(n_jobs=NJ)(delayed(_gts)(views[OBS[i]]) for i in ALL)))
NGT = {i: [len(g) for g in GT[i]] for i in ALL}
print(f'GT ready {time.time() - t0:.0f}s', flush=True)
CFGV = ('faint', 0.3)

def combo_map(i):
    acc = np.zeros((2, H, W), np.float32)
    for m in MEMBERS: acc += np.load(PATHS[m], mmap_mode='r')[i]
    return np.round(acc / len(MEMBERS)).astype(np.uint8)

def one(i, img, gts_all):
    P = combo_map(i)
    sets, dist_big, n_big = candidate_sets(P, img); out = {}
    for src_ in ['B', 0.3]:
        lab = sets[src_]
        X = cand_features(lab, P, img, dist_big, n_big, 0.6 if src_ == 'B' else src_)
        IOU, GID = match_table(lab, gts_all)
        out[src_] = dict(X=X, IOU=IOU, GID=GID, png=cv2.imencode('.png', lab.astype(np.uint16))[1].tobytes())
    out['prob'] = cv2.imencode('.png', P[0])[1].tobytes()          # ensemble mask probability (uint8)
    return out

C = dict(zip(ALL, Parallel(n_jobs=NJ)(delayed(one)(i, DATA[i][0], GT[i]) for i in ALL)))
print(f'candidates ready {time.time() - t0:.0f}s', flush=True)
own = np.concatenate([np.full(len(pool(i, CFGV)[0]), i) for i in ALL])
X = np.concatenate([pool(i, CFGV)[0] for i in ALL]); y = np.concatenate([(pool(i, CFGV)[1] > 0.5).mean(1) for i in ALL])
gbm = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
sc = np.zeros(len(y))
for k in range(5):
    tr, va = folds[own] != k, folds[own] == k
    sc[va] = gbm().fit(X[tr], y[tr]).predict(X[va])
SC = {i: sc[own == i] for i in ALL}
sweep = {s: pq_keep(CFGV, lambda i, X_, s=s: SC[i] >= s)[0] for s in [0.25, 0.3, 0.35, 0.4, 0.45]}
s_best = max(sweep, key=sweep.get)
pq_all, S = pq_keep(CFGV, lambda i, X_: SC[i] >= s_best)
print('filter sweep', {k: round(v, 4) for k, v in sweep.items()}, '-> keep s >=', s_best, f'| OOF PQ {pq_all:.4f} TP/FP/FN {S[1:].astype(int)}', flush=True)

def gt_polys(view):                       # polygons in the same order as gt_view (empty masks skipped)
    out = []
    for poly, sp in view:
        if poly_mask(poly).any(): out.append((poly.astype(np.float32), np.asarray(sp, np.float32)))
    return out

REC = []
for i in ALL:
    b = C[i]['B']; keepB = np.exp(b['X'][:, 0]) >= 400
    srcs = [('B', j + 1) for j in np.where(keepB)[0]] + [(0.3, j + 1) for j in range(len(C[i][0.3]['X']))]
    Xp, IOU, GID = pool(i, CFGV); kept = SC[i] >= s_best
    nv = len(NGT[i]); gt_match = [np.zeros(n, bool) for n in NGT[i]]; pred_match = np.zeros((len(kept), nv), bool)
    view_S = np.zeros((nv, 4))
    for v in range(nv):                   # same bookkeeping as pq_keep: best kept prediction per GT, IoU > 0.5
        best = {}
        for r in np.where(kept & (IOU[:, v] > 0.5))[0]:
            g = GID[r, v]
            if IOU[r, v] > best.get(g, (0, -1))[0]: best[g] = (IOU[r, v], r)
        for g, (iou, r) in best.items(): gt_match[v][g] = True; pred_match[r, v] = True
        tp = len(best); view_S[v] = [sum(x[0] for x in best.values()), tp, kept.sum() - tp, NGT[i][v] - tp]
    REC.append(dict(i=int(i), fn=OBS[i], site=OBS[i][14:16], fold=int(folds[i]), n_views=nv,
                    B_png=C[i]['B']['png'], F_png=C[i][0.3]['png'], prob_png=C[i]['prob'],
                    cand_src=srcs, X=Xp.astype(np.float32), area=np.exp(Xp[:, 0]), mean_p=Xp[:, 1], contrast=Xp[:, 7], score=SC[i], kept=kept,
                    iou=IOU, gid=GID, pred_match=pred_match, gt_match=gt_match, view_S=view_S,
                    gt=[gt_polys(v) for v in views[OBS[i]]]))
os.makedirs('/root/autodl-tmp/runs/vis', exist_ok=True)
pickle.dump(dict(rec=REC, s_best=s_best, pq=pq_all, S=S, feats=FEATS), open('/root/autodl-tmp/runs/vis/vis.pkl', 'wb'))
print('VIS DONE', f'{time.time() - t0:.0f}s', os.path.getsize('/root/autodl-tmp/runs/vis/vis.pkl') / 1e6, 'MB', flush=True)
