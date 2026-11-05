import sys, itertools
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
MEMBERS = sys.argv_members if hasattr(sys, 'argv_members') else ['cnxt', 'effb4', 'regy']
import os as _os
MEMBERS = _os.environ.get('MEMBERS', 'cnxt,effb4,regy').split(',')
PATHS = {m: f'/root/autodl-tmp/runs/m_{m}/oof_maps.npy' for m in MEMBERS}
t0 = time.time()
GT = {i: [gt_view(v) for v in views[OBS[i]]] for i in ALL}
NGT = {i: [len(g) for g in GT[i]] for i in ALL}
print(f'GT ready {time.time() - t0:.0f}s | members {MEMBERS}', flush=True)
BPP = dict(thr=0.5, min_area=400, lo=None, link=40)

def combo_map(i, w):
    acc = np.zeros((2, H, W), np.float32)
    for m, wm in zip(MEMBERS, w):
        if wm > 0: acc += wm * np.load(PATHS[m], mmap_mode='r')[i]
    return np.round(acc / sum(w)).astype(np.uint8)

def img_stats(i, w, gts_all):
    lab = instances('B', BPP, combo_map(i, w)); parea = np.bincount(lab.ravel(), minlength=lab.max() + 1)
    S = np.zeros(4)
    for gts in gts_all:
        S += match_view(lab, parea, gts)[:4]
    return S
def pq(S): return S[0] / max(S[1] + 0.5 * S[2] + 0.5 * S[3], 1)

# ---- stage 1: weight grid (step 0.1) scored with post-processing B, per fold ----
grid = [w for w in itertools.product(*[np.round(np.arange(0, 1.01, 0.1), 1)] * len(MEMBERS)) if abs(sum(w) - 1) < 1e-6]
if len(MEMBERS) > 3:
    grid = [w for w in itertools.product(*[np.round(np.arange(0, 1.01, 0.2), 1)] * len(MEMBERS)) if abs(sum(w) - 1) < 1e-6]
eq = tuple([round(1 / len(MEMBERS), 4)] * len(MEMBERS)); grid = [eq] + [w for w in grid if w != eq]
FS = {}
t0 = time.time()
for w in grid:
    S = Parallel(n_jobs=48)(delayed(img_stats)(i, w, GT[i]) for i in ALL)
    FS[w] = np.array([sum(S[i] for i in np.where(folds == k)[0]) for k in range(5)])
    print(f'{w}: B PQ {pq(FS[w].sum(0)):.4f}  folds {[round(pq(f), 4) for f in FS[w]]}  [{time.time() - t0:.0f}s]', flush=True)
rank = sorted(grid, key=lambda w: -pq(FS[w].sum(0)))
print('\ntop 10 by B PQ:'); [print(w, round(pq(FS[w].sum(0)), 4)) for w in rank[:10]]
# honest (nested) estimate: choose weights on 4 folds, score the 5th
held = np.zeros(4)
for k in range(5):
    others = [j for j in range(5) if j != k]
    wb = max(grid, key=lambda w: pq(FS[w][others].sum(0)))
    held += FS[wb][k]; print(f'  fold {k}: weights chosen on other folds {wb} -> held-out B PQ {pq(FS[wb][k]):.4f} (equal {pq(FS[eq][k]):.4f})')
print(f'nested B PQ {pq(held):.4f} vs equal weights {pq(FS[eq].sum(0)):.4f}', flush=True)

# ---- stage 2: full pipeline (candidates + FP filter) for equal weights and the top 3 ----
def cand_tables(i, w, img, gts_all):
    P = combo_map(i, w)
    sets, dist_big, n_big = candidate_sets(P, img); out = {}
    for src_, lab in sets.items():
        X = cand_features(lab, P, img, dist_big, n_big, 0.6 if src_ == 'B' else src_)
        IOU, GID = match_table(lab, gts_all); out[src_] = dict(X=X, IOU=IOU, GID=GID)
    return out
gbm = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
for w in [eq] + [x for x in rank[:3] if x != eq]:
    t0 = time.time()
    C = dict(zip(ALL, Parallel(n_jobs=48)(delayed(cand_tables)(i, w, DATA[i][0], GT[i]) for i in ALL)))
    res = []
    for cfg in [('min_area', 150), ('min_area', 250), ('faint', 0.5), ('faint', 0.4), ('faint', 0.3)]:
        own = np.concatenate([np.full(len(pool(i, cfg)[0]), i) for i in ALL])
        X = np.concatenate([pool(i, cfg)[0] for i in ALL]); y = np.concatenate([(pool(i, cfg)[1] > 0.5).mean(1) for i in ALL])
        sc = np.zeros(len(y))
        for k in range(5):
            tr, va = folds[own] != k, folds[own] == k
            sc[va] = gbm().fit(X[tr], y[tr]).predict(X[va])
        SCx = {i: sc[own == i] for i in ALL}
        sweep = {s: pq_keep(cfg, lambda i, X_, s=s: SCx[i] >= s)[0] for s in [0.25, 0.3, 0.35, 0.4, 0.45]}
        sb = max(sweep, key=sweep.get)
        pf = [round(pq_keep(cfg, lambda i, X_: SCx[i] >= sb, np.where(folds == k)[0])[0], 4) for k in range(5)]
        res.append((cfg, round(sweep[sb], 4), pf))
    best = max(res, key=lambda r: r[1])
    print(f'FULL {w}: best {best[0]} PQ {best[1]} folds {best[2]} | all: ' + ', '.join(f'{c}:{p}' for c, p, _ in res) + f' [{time.time() - t0:.0f}s]', flush=True)
print('WSEARCH DONE', flush=True)
