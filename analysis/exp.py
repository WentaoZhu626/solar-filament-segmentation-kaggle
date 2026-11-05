import sys
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
ALL = np.arange(len(OBS))
R = '/root/autodl-tmp/runs/err3/'
OOFM = np.load(R + 'oof_maps.npy', mmap_mode='r')
ST = pickle.load(open(R + 'err/state.pkl', 'rb')); BC = ST['BEST_C']; SC0 = ST['RESF'][BC]['SC']; S0 = ST['RESF'][BC]['s']
t0 = time.time()
GT = {i: [gt_view(v) for v in views[OBS[i]]] for i in ALL}
NGT = {i: [len(g) for g in GT[i]] for i in ALL}
print(f'GT ready {time.time() - t0:.0f}s | best config {BC} keep >= {S0}', flush=True)

def pqfolds(make_lab):
    rk = [evaluate_inst(make_lab, np.where(folds == k)[0]) for k in range(5)]
    tp, fp, fn = (sum(x[m] for x in rk) for m in ('TP', 'FP', 'FN')); s = sum(x['SQ'] * x['TP'] for x in rk)
    return s / (tp + 0.5 * fp + 0.5 * fn), [round(x['PQ'], 4) for x in rk], (tp, fp, fn)

def kept_labels(P, img, SCi, s, bcfg):
    sets, _, _ = candidate_sets(P, img)
    labB = sets['B']; aB = np.bincount(labB.ravel(), minlength=labB.max() + 1)[1:]
    idsB = np.where(aB >= (bcfg[1] if bcfg[0] == 'min_area' else 400))[0] + 1
    out = np.zeros((H, W), np.int32); n = 0; keep = SCi >= s
    for j, k in enumerate(idsB):
        if keep[j]: n += 1; out[labB == k] = n
    if bcfg[0] == 'faint':
        labE = sets[bcfg[1]]
        for j, k in enumerate(range(1, labE.max() + 1)):
            if keep[len(idsB) + j]: n += 1; out[labE == k] = n
    return out

# ---------------- part 1: thinner SMALL filaments (same kept instances, only their masks change) ----------------
t0 = time.time()
LAB = dict(zip(ALL, Parallel(n_jobs=48)(delayed(kept_labels)(np.array(OOFM[i]), DATA[i][0], SC0[i], S0, BC) for i in ALL)))
print(f'kept label maps {time.time() - t0:.0f}s', flush=True)
base = pqfolds(lambda i: LAB[i]); print('baseline (ens3)', round(base[0], 4), base[1], base[2], flush=True)
K3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
def thin(i, A0, t, mode):
    lab = LAB[i]; area = np.bincount(lab.ravel(), minlength=lab.max() + 1); small = area < A0; small[0] = False
    if not small[1:].any(): return lab
    sm = small[lab]
    if mode == 'thr':
        drop = sm & (OOFM[i][0] <= t * 255)
    else:                                       # erode each small instance by 1 px
        er = cv2.erode((sm).astype(np.uint8), K3)
        drop = sm & (er == 0)
    rem = np.bincount(lab[sm & ~drop], minlength=len(area))
    ok = rem >= 0.25 * area                     # never let an instance vanish / shrink below 25 %
    drop &= ok[lab]
    out = lab.copy(); out[drop] = 0; return out
rows = []
for A0 in [400, 800, 1500, 3000]:
    for mode, t in [('thr', 0.6), ('thr', 0.7), ('thr', 0.8), ('erode', 0)]:
        r = pqfolds(lambda i: thin(i, A0, t, mode))
        rows.append(dict(small_below=A0, variant=f'{mode} {t}' if mode == 'thr' else 'erode 1px', PQ=round(r[0], 4),
                         d=round(r[0] - base[0], 4), folds_better=int(sum(a > b for a, b in zip(r[1], base[1]))), TP=r[2][0], FP=r[2][1], FN=r[2][2], per_fold=r[1]))
        print(rows[-1], flush=True)
print(pd.DataFrame(rows).to_string(index=False), flush=True)

# ---------------- part 2: larger spine-link distance for B (joins fragments across longer faint gaps) ----------------
def cand_tables(P, img, gts):
    sets, dist_big, n_big = candidate_sets(P, img); out = {}
    for src_, lab in sets.items():
        X = cand_features(lab, P, img, dist_big, n_big, 0.6 if src_ == 'B' else src_)
        IOU, GID = match_table(lab, gts); out[src_] = dict(X=X, IOU=IOU, GID=GID)
    return out
gbm = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
for L in [40, 70, 100, 150]:
    B_CFG['link'] = L; t0 = time.time()
    C = dict(zip(ALL, Parallel(n_jobs=48)(delayed(cand_tables)(np.array(OOFM[i]), DATA[i][0], GT[i]) for i in ALL)))
    res = []
    for cfg in [BC, ('min_area', 150), ('faint', 0.4)]:
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
        res.append((cfg, round(sweep[sb], 4), sb, pf))
    print(f'link {L}: ' + ' | '.join(f'{c} PQ {p} (s {s}) folds {f}' for c, p, s, f in res) + f'  [{time.time() - t0:.0f}s]', flush=True)
print('EXP DONE', flush=True)
