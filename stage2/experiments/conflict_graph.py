
# Conflict-graph selection over heterogeneous hypotheses (B instances, faint blobs, plain threshold components,
# refined masks).  Each hypothesis gets an out-of-fold score y_hat (share of annotators matching); keep a set of
# pairwise non-overlapping hypotheses maximising sum(q_bar * y_hat - tau) (max-weight independent set, exact MILP),
# tau = PQ/2 by Dinkelbach iteration.  Nested threshold components alone reproduce the component-tree special case.
import os, sys, time, pickle
STAGE = sys.argv[1]
OUTD = os.environ.get('CGDIR', '/root/cg/'); os.makedirs(OUTD, exist_ok=True)
if STAGE == 'build':
    src = open('/root/autodl-tmp/analysis/vis_prep2.py').read()
    exec(src[:src.index('def one(i, img, gts_all):')])        # GT, DATA, combo_map, candidate_sets, cand_features, match_table ...
    RD = os.environ.get('REFDIR', '/root/refine/'); META = pickle.load(open(RD + 'meta.pkl', 'rb')); PRED = np.load(RD + 'pred.npy', mmap_mode='r')
    byimg = {}
    for c, m in enumerate(META): byimg.setdefault(int(m[0]), []).append(c)
    TS = [0.3, 0.4, 0.5, 0.6, 0.7]; MIN_T = 100
    def build(i, img, gts, refs):
        P = combo_map(i)
        sets, dist_big, n_big = candidate_sets(P, img)
        labB0 = sets['B']; aB = np.bincount(labB0.ravel(), minlength=labB0.max() + 1)[1:]
        keepB = np.where(aB >= 400)[0] + 1
        mapB = np.zeros(labB0.max() + 1, np.int32); mapB[keepB] = np.arange(1, len(keepB) + 1)
        labB = mapB[labB0]; labF = sets[0.3]
        srcs = [('B', labB, 0.6), ('F', labF, 0.3)]
        # parent candidates (vis / CNN order): B then F
        nB, nF = labB.max(), labF.max()
        # refined masks of every B/F candidate, overlaps resolved by original area
        par_area = np.concatenate([np.bincount(labB.ravel(), minlength=nB + 1)[1:], np.bincount(labF.ravel(), minlength=nF + 1)[1:]])
        labR = np.zeros((H, W), np.int32); rpar = []
        order = sorted(refs, key=lambda r: -par_area[r[0]] if r[0] < len(par_area) else 0)
        for (j, oy, ox, side, pr) in order:
            p = cv2.resize(pr, (side, side), interpolation=cv2.INTER_LINEAR) >= 128
            full = np.zeros((H, W), bool); ya, xa = max(oy, 0), max(ox, 0); yb, xb = min(oy + side, H), min(ox + side, W)
            full[ya:yb, xa:xb] = p[ya - oy:yb - oy, xa - ox:xb - ox]
            full &= labR == 0
            if full.sum() < 30: continue
            rpar.append(j); labR[full] = len(rpar)
        srcs.append(('R', labR, 0.5))
        for t in TS:
            n, lab, st, _ = cv2.connectedComponentsWithStats((P[0] > t * 255).astype(np.uint8), connectivity=8)
            keep = np.zeros(n, np.int32); ok = np.where(st[1:, cv2.CC_STAT_AREA] >= MIN_T)[0] + 1
            keep[ok] = np.arange(1, len(ok) + 1); srcs.append((f'T{t}', keep[lab], t))
        parent = np.concatenate([labB + 0, np.where(labF > 0, labF + nB, 0)]) if False else None
        parlab = np.where(labB > 0, labB, np.where(labF > 0, labF + nB, 0))     # B/F are disjoint by construction
        Xs, IOUs, GIDs, SRC, AREA, PAR, PIOU, LABS = [], [], [], [], [], [], [], []
        for k, (name, lab, st_) in enumerate(srcs):
            n = int(lab.max())
            if n == 0: LABS.append(lab); continue
            X = cand_features(lab, P, img, dist_big, n_big, st_); IOU, GID = match_table(lab, gts)
            area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
            m = (lab > 0) & (parlab > 0)
            jo = np.bincount(lab[m].astype(np.int64) * (nB + nF + 1) + parlab[m], minlength=(n + 1) * (nB + nF + 1)).reshape(n + 1, nB + nF + 1)[1:, 1:]
            if name == 'R':
                par = np.array(rpar)
            else:
                par = np.where(jo.max(1) > 0, jo.argmax(1), -1) if jo.size else np.full(n, -1)
            inter = np.array([jo[a, par[a]] if par[a] >= 0 else 0 for a in range(n)])
            piou = np.array([inter[a] / (area[a] + par_area[par[a]] - inter[a]) if par[a] >= 0 else 0 for a in range(n)])
            Xs.append(X); IOUs.append(IOU); GIDs.append(GID); SRC += [k] * n; AREA.append(area); PAR.append(par); PIOU.append(piou); LABS.append(lab)
        off = np.cumsum([0] + [int(l.max()) for l in LABS])
        edges = set()
        for a in range(len(LABS)):
            for b in range(a + 1, len(LABS)):
                la, lb = LABS[a], LABS[b]; na, nb = int(la.max()), int(lb.max())
                if na == 0 or nb == 0: continue
                m = (la > 0) & (lb > 0)
                if not m.any(): continue
                pairs = np.unique(la[m].astype(np.int64) * (nb + 1) + lb[m])
                for pv in pairs: edges.add((off[a] + int(pv // (nb + 1)) - 1, off[b] + int(pv % (nb + 1)) - 1))
        return dict(X=np.concatenate(Xs), IOU=np.concatenate(IOUs), GID=np.concatenate(GIDs), src=np.array(SRC), area=np.concatenate(AREA),
                    par=np.concatenate(PAR), piou=np.concatenate(PIOU), edges=np.array(sorted(edges), np.int32).reshape(-1, 2), nB=int(nB), nF=int(nF),
                    labs=[cv2.imencode('.png', l.astype(np.uint16))[1].tobytes() for l in LABS], prob=cv2.imencode('.png', P[0])[1].tobytes(), i=int(i))
    t0 = time.time()
    def refs_of(i):
        return [(int(META[c][1]), int(META[c][4]), int(META[c][5]), int(META[c][6]), np.array(PRED[c])) for c in byimg.get(int(i), [])]
    out = Parallel(n_jobs=int(os.environ.get('NJ', 14)), batch_size=1)(delayed(build)(i, DATA[i][0], GT[i], refs_of(i)) for i in ALL)
    pickle.dump(dict(H=out, folds=folds, NGT=NGT, src_names=['B', 'F', 'R'] + [f'T{t}' for t in TS]), open(OUTD + os.environ.get('HYPF', 'hyp.pkl'), 'wb'))
    print(f'BUILD DONE {time.time() - t0:.0f}s, hypotheses {sum(len(h["src"]) for h in out)}, edges {sum(len(h["edges"]) for h in out)}', flush=True)

if STAGE == 'select':
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from joblib import Parallel, delayed
    D = pickle.load(open(OUTD + os.environ.get('HYPF', 'hyp.pkl'), 'rb')); HY = D['H']; folds = np.asarray(D['folds']); NGT = D['NGT']; SN = D['src_names']
    V = pickle.load(open('/root/autodl-tmp/runs/vis/vis.pkl', 'rb'))
    cnt = [len(r['kept']) for r in V['rec']]; ours = [h['nB'] + h['nF'] for h in HY]
    print('B+F counts match vis:', cnt == ours, sum(cnt), sum(ours), flush=True)
    CO = '/root/autodl-tmp/runs/cnnf/'
    cn_all = np.mean([np.load(CO + f'oof{t}.npy') for t in ['', '_cnxt', '_effb3', '_r50', '_r34_192']], 0)
    offs = np.cumsum([0] + cnt)
    USE_CNN = os.environ.get('USE_CNN', '1') == '1'
    if os.environ.get('CGCNN'):
        _o = np.load(OUTD + 'cnn_oof.npy'); _c = np.cumsum([0] + [len(h['src']) for h in HY]); CG = [_o[_c[i]:_c[i + 1]] for i in range(len(HY))]
    feats, ys, own = [], [], []
    for i, h in enumerate(HY):
        n = len(h['src']); cn = cn_all[offs[i]:offs[i + 1]] if cnt[i] == ours[i] else np.zeros(ours[i])
        pc = np.where(h['par'] >= 0, cn[np.clip(h['par'], 0, max(len(cn) - 1, 0))] if len(cn) else -1, -1)
        oh = np.eye(len(SN))[h['src']]
        f = [h['X'], oh, h['piou'][:, None]] + ([pc[:, None]] if USE_CNN else [])
        if os.environ.get('CGCNN'): f.append(CG[i])
        feats.append(np.hstack(f)); ys.append(np.where(h['IOU'] > 0.5, h['IOU'], 0).mean(1) / 0.6722 if os.environ.get('TARGET') == 'u' else (h['IOU'] > 0.5).mean(1)); own.append(np.full(n, i))
    F = np.vstack(feats); y = np.concatenate(ys); own = np.concatenate(own); fo = folds[own]
    yh = np.zeros(len(y))
    for k in range(5):
        tr, te = fo != k, fo == k
        yh[te] = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(F[tr], y[tr]).predict(F[te])
    m_iou = np.concatenate([h['IOU'][h['IOU'] > 0.5] for h in HY]); qbar = float(m_iou.mean())
    print(f'hypotheses {len(y)} | qbar {qbar:.4f} | per source n:', {SN[s]: int((np.concatenate([h["src"] for h in HY]) == s).sum()) for s in range(len(SN))}, flush=True)
    if os.environ.get('PERSRC') == '1':                   # separate scorer per source family (B/F, R, T)
        fam = np.concatenate([h['src'] for h in HY]); fam = np.where(fam <= 1, 0, np.where(fam == 2, 1, 2))
        for k in range(5):
            for f_ in range(3):
                tr, te = (fo != k) & (fam == f_), (fo == k) & (fam == f_)
                yh[te] = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(F[tr], y[tr]).predict(F[te])
    if os.environ.get('ORACLE') == '1':                   # upper bound: true expected matched IoU as weight
        yh = np.concatenate([np.where(h['IOU'] > 0.5, h['IOU'], 0).mean(1) for h in HY]) / qbar
    YH = [yh[own == i] for i in range(len(HY))]
    def choose(i, allowed, tau):
        h = HY[i]; w = qbar * YH[i] - tau
        cand = np.where(np.isin(h['src'], allowed) & (w > 0))[0]
        if len(cand) == 0: return cand
        pos = -np.ones(len(h['src']), int); pos[cand] = np.arange(len(cand))
        E = h['edges']; E = E[(pos[E[:, 0]] >= 0) & (pos[E[:, 1]] >= 0)] if len(E) else E
        if len(E) == 0: return cand
        a, b = pos[E[:, 0]], pos[E[:, 1]]
        nc, comp = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(len(cand), len(cand))), directed=False)
        keep = []
        for c in range(nc):
            idx = np.where(comp == c)[0]
            if len(idx) == 1: keep.append(cand[idx[0]]); continue
            loc = -np.ones(len(cand), int); loc[idx] = np.arange(len(idx))
            sel = (loc[a] >= 0)
            ea, eb = loc[a[sel]], loc[b[sel]]
            A = np.zeros((len(ea), len(idx))); A[np.arange(len(ea)), ea] = 1; A[np.arange(len(ea)), eb] = 1
            r = milp(-w[cand[idx]], constraints=LinearConstraint(A, -np.inf, 1), integrality=np.ones(len(idx)), bounds=Bounds(0, 1))
            keep += list(cand[idx[np.round(r.x) > 0.5]])
        return np.array(sorted(keep), int)
    def score(i, ch):
        h = HY[i]; S = np.zeros(4)
        for v in range(h['IOU'].shape[1]):
            best = {}
            for c in ch:
                if h['IOU'][c, v] > 0.5:
                    g = h['GID'][c, v]
                    if h['IOU'][c, v] > best.get(g, 0): best[g] = h['IOU'][c, v]
            tp = len(best); S += [sum(best.values()), tp, len(ch) - tp, NGT[i][v] - tp]
        return S
    def run(allowed, tau):
        Ss = [score(i, choose(i, allowed, tau)) for i in range(len(HY))]
        S = np.sum(Ss, 0); return S[0] / (S[1] + 0.5 * S[2] + 0.5 * S[3]), S, np.array(Ss)
    VARS = {'B+F (threshold rule)': [0, 1], 'T only (component tree)': [3, 4, 5, 6, 7], 'B+F+T': [0, 1, 3, 4, 5, 6, 7],
            'B+F+R': [0, 1, 2], 'R only': [2], 'ALL': list(range(len(SN)))}
    def one_var(name, al):
        lines = []; tau = 0.22
        for it in range(4):
            p, S, _ = run(al, tau); lines.append(f'  Dinkelbach it{it} tau {tau:.4f} -> PQ {p:.4f} TP/FP/FN {S[1:].astype(int)}'); tau = p / 2
        p, S, Ss = run(al, tau)
        pf = []
        for k in range(5):
            Sk = Ss[folds[:len(HY)] == k].sum(0); pf.append(round(Sk[0] / (Sk[1] + 0.5 * Sk[2] + 0.5 * Sk[3]), 4))
        scan = {t: round(run(al, t)[0], 4) for t in [0.18, 0.22, 0.26, 0.30]}
        return name, tau, p, S, pf, scan, lines, Ss
    R = Parallel(n_jobs=len(VARS))(delayed(one_var)(nm, al) for nm, al in VARS.items())
    for name, tau, p, S, pf, scan, lines, Ss in R:
        print(f'== {name}: tau* {tau:.4f} PQ {p:.4f} TP/FP/FN {S[1:].astype(int)} per fold {pf} | scan {scan}', flush=True)
        for l in lines: print(l)
    pickle.dump({r[0]: r[7] for r in R}, open(OUTD + f'per_img_sel{os.environ.get("TAGX", "")}.pkl', 'wb'))
    print('SELECT DONE')

if STAGE == 'crops':
    import numpy as np, cv2
    from joblib import Parallel, delayed
    S = 128
    D = pickle.load(open(OUTD + os.environ.get('HYPF', 'hyp2.pkl'), 'rb')); HY = D['H']
    IMG = np.load('/root/autodl-tmp/cache/img.npy', mmap_mode='r')
    def crops(h, img):
        dec = lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
        labs = [dec(b) for b in h['labs']]; prob = dec(h['prob'])
        base = ((labs[0] > 0) | (labs[1] > 0))
        out = []
        for lab in labs:
            for lid in range(1, int(lab.max()) + 1):
                m = lab == lid
                ys, xs = np.nonzero(m); cy, cx = ys.mean(), xs.mean()
                side = max(max(ys.max() - ys.min(), xs.max() - xs.min()) * 1.5 + 32, 64)
                def cut(a, sd, fill, interp):
                    hh = int(sd // 2); x0, y0 = int(cx) - hh, int(cy) - hh; pad = hh + 2
                    ap = cv2.copyMakeBorder(a, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=fill)
                    return cv2.resize(ap[y0 + pad:y0 + pad + 2 * hh, x0 + pad:x0 + pad + 2 * hh], (S, S), interpolation=interp)
                mm = m.astype(np.uint8) * 255; oth = (base & ~m).astype(np.uint8) * 255
                out.append(np.stack([cut(img, side, 128, cv2.INTER_AREA), cut(prob, side, 0, cv2.INTER_AREA), cut(mm, side, 0, cv2.INTER_AREA),
                                     cut(img, 512, 128, cv2.INTER_AREA), cut(prob, 512, 0, cv2.INTER_AREA), cut(mm, 512, 0, cv2.INTER_AREA), cut(oth, 512, 0, cv2.INTER_AREA)]))
        return np.stack(out) if out else np.zeros((0, 7, S, S), np.uint8)
    t0 = time.time()
    Xs = Parallel(n_jobs=14, batch_size=2)(delayed(crops)(h, np.asarray(IMG[h['i']])) for h in HY)
    n = sum(len(x) for x in Xs); assert n == sum(len(h['src']) for h in HY), (n,)
    X = np.lib.format.open_memmap(OUTD + 'crops.npy', mode='w+', dtype=np.uint8, shape=(n, 7, S, S)); o = 0
    for x in Xs: X[o:o + len(x)] = x; o += len(x)
    X.flush(); print('CROPS DONE', X.shape, f'{time.time() - t0:.0f}s', flush=True)

if STAGE == 'cnn':
    import numpy as np, torch, torch.nn as nn, timm
    D = pickle.load(open(OUTD + os.environ.get('HYPF', 'hyp2.pkl'), 'rb')); HY = D['H']; folds = np.asarray(D['folds'])
    y = np.concatenate([(h['IOU'] > 0.5).mean(1) for h in HY]).astype(np.float32)
    u = np.concatenate([np.where(h['IOU'] > 0.5, h['IOU'], 0).mean(1) for h in HY]).astype(np.float32)
    fold = np.concatenate([np.full(len(h['src']), folds[k]) for k, h in enumerate(HY)])
    X = np.load(OUTD + 'crops.npy', mmap_mode='r'); DEV = 'cuda'; EP = int(os.environ.get('EP', 12)); BS = 128
    Xt = torch.from_numpy(np.ascontiguousarray(X)).to(DEV); T = torch.from_numpy(np.stack([y, u], 1)).to(DEV)
    oof = np.zeros((len(y), 2), np.float32); t0 = time.time()
    def aug(x):
        if np.random.rand() < 0.5: x = x.flip(-1)
        if np.random.rand() < 0.5: x = x.flip(-2)
        return torch.rot90(x, np.random.randint(4), (2, 3))
    for k in range(5):
        tr = np.where(fold != k)[0]; va = np.where(fold == k)[0]
        model = timm.create_model('resnet34', pretrained=True, in_chans=7, num_classes=2, drop_rate=0.2).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2); steps = EP * ((len(tr) + BS - 1) // BS)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.1)
        for ep in range(EP):
            model.train(); perm = np.random.permutation(tr); tl = 0
            for b in range(0, len(perm), BS):
                idx = torch.from_numpy(perm[b:b + BS]).to(DEV); x = aug(Xt[idx].float() / 255)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = nn.functional.binary_cross_entropy_with_logits(model(x).float(), T[idx])
                opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tl += loss.item() * len(idx)
            print(f'fold {k} ep {ep} loss {tl / len(tr):.4f} {time.time() - t0:.0f}s', flush=True)
        model.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for b in range(0, len(va), 512):
                x = Xt[torch.from_numpy(va[b:b + 512]).to(DEV)].float() / 255
                oof[va[b:b + 512]] = (sum(torch.sigmoid(model(torch.rot90(x, q, (2, 3))).float()) for q in range(4)) / 4).cpu().numpy()
        torch.save(model.state_dict(), OUTD + f'cgcnn_fold{k}.pth'); del model; torch.cuda.empty_cache()
    np.save(OUTD + 'cnn_oof.npy', oof); print('CNN DONE', f'{time.time() - t0:.0f}s')
