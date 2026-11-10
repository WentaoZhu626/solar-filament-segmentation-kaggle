
# Their pipeline in our setting: component tree over threshold levels (T sources only) -> scorer u_hat -> exact
# max-weight antichain (MILP == tree DP) with tau = PQ/2 -> stage-2 refiner (coco) on the selected nodes.
import os, pickle, time, numpy as np, cv2, torch
import segmentation_models_pytorch as smp
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
CGD = '/root/cg3/'
D = pickle.load(open(CGD + 'hyp2.pkl', 'rb')); HY = D['H']; folds = np.asarray(D['folds']); NGT = D['NGT']; SN = D['src_names']
V = pickle.load(open('/root/autodl-tmp/runs/vis/vis.pkl', 'rb')); cnt = [len(r['kept']) for r in V['rec']]; offs = np.cumsum([0] + cnt)
CO = '/root/autodl-tmp/runs/cnnf/'; cn_all = np.mean([np.load(CO + f'oof{t}.npy') for t in ['', '_cnxt', '_effb3', '_r50', '_r34_192']], 0)
O = np.load(CGD + 'cnn_oof.npy'); co = np.cumsum([0] + [len(h['src']) for h in HY])
TSRC = [i for i, n in enumerate(SN) if n.startswith('T')]
feats, ys, own = [], [], []
for i, h in enumerate(HY):
    cn = cn_all[offs[i]:offs[i + 1]]
    pc = np.where(h['par'] >= 0, cn[np.clip(h['par'], 0, max(len(cn) - 1, 0))] if len(cn) else -1, -1)
    feats.append(np.hstack([h['X'], np.eye(len(SN))[h['src']], h['piou'][:, None], pc[:, None], O[co[i]:co[i + 1]]]))
    ys.append(np.where(h['IOU'] > 0.5, h['IOU'], 0).mean(1)); own.append(np.full(len(h['src']), i))
F = np.vstack(feats); u = np.concatenate(ys); own = np.concatenate(own); fo = folds[own]
isT = np.isin(np.concatenate([h['src'] for h in HY]), TSRC)
MODE = os.environ.get('TRAIN_ON', 'T')          # 'T': scorer trained on tree nodes only (their setting); 'ALL': on all hypotheses
uh = np.zeros(len(u))
for k in range(5):
    tr = (fo != k) & (isT if MODE == 'T' else True); te = fo == k
    uh[te] = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(F[tr], u[tr]).predict(F[te])
UH = [uh[own == i] for i in range(len(HY))]
def choose(i, tau):
    h = HY[i]; w = UH[i] - tau; cand = np.where(np.isin(h['src'], TSRC) & (w > 0))[0]
    if len(cand) == 0: return cand
    pos = -np.ones(len(h['src']), int); pos[cand] = np.arange(len(cand)); E = h['edges']
    E = E[(pos[E[:, 0]] >= 0) & (pos[E[:, 1]] >= 0)] if len(E) else E
    if len(E) == 0: return cand
    a, b = pos[E[:, 0]], pos[E[:, 1]]
    nc, comp = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(len(cand),) * 2), directed=False); keep = []
    for c in range(nc):
        idx = np.where(comp == c)[0]
        if len(idx) == 1: keep.append(cand[idx[0]]); continue
        loc = -np.ones(len(cand), int); loc[idx] = np.arange(len(idx)); sel = loc[a] >= 0; ea, eb = loc[a[sel]], loc[b[sel]]
        A = np.zeros((len(ea), len(idx))); A[np.arange(len(ea)), ea] = 1; A[np.arange(len(ea)), eb] = 1
        r = milp(-w[cand[idx]], constraints=LinearConstraint(A, -np.inf, 1), integrality=np.ones(len(idx)), bounds=Bounds(0, 1))
        keep += list(cand[idx[np.round(r.x) > 0.5]])
    return np.array(sorted(keep), int)
def score_idx(i, ch):
    h = HY[i]; S = np.zeros(4)
    for v in range(h['IOU'].shape[1]):
        best = {}
        for c in ch:
            if h['IOU'][c, v] > 0.5: g = h['GID'][c, v]; best[g] = max(best.get(g, 0), h['IOU'][c, v])
        tp = len(best); S += [sum(best.values()), tp, len(ch) - tp, NGT[i][v] - tp]
    return S
pq = lambda S: S[0] / (S[1] + 0.5 * S[2] + 0.5 * S[3])
tau = 0.22
for it in range(4):
    S = sum(score_idx(i, choose(i, tau)) for i in range(len(HY))); print(f'tree stage 1: tau {tau:.4f} -> PQ {pq(S):.4f} TP/FP/FN {S[1:].astype(int)}', flush=True); tau = pq(S) / 2
CH = [choose(i, tau) for i in range(len(HY))]
S1 = np.array([score_idx(i, CH[i]) for i in range(len(HY))])
# ---- stage 2: coco refiner (fold-k model for fold-k images) on the selected tree nodes ----
from pycocotools import mask as mu
src = open('/root/autodl-tmp/run_ens.py').read()
exec(src[src.index('def poly_mask(poly):'):src.index('def as_xy(a):')].replace('H, W', '2048, 2048'))
VREC = V['rec']
IMG = np.load('/root/autodl-tmp/cache/img.npy', mmap_mode='r'); R_ = 256; dev = 'cuda'
models = []
for k in range(5):
    m = smp.Unet('resnet34', encoder_weights=None, in_channels=4, classes=1).to(dev).eval(); m.load_state_dict(torch.load(f'/root/refine_coco/ref_fold{k}.pth', map_location=dev)); models.append(m)
def geom(m):
    ys, xs = np.where(m); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(max(y1 - y0, x1 - x0) * 1.3 + 48); return (y0 + y1) // 2 - side // 2, (x0 + x1) // 2 - side // 2, side
def crop(a, oy, ox, side):
    o = np.zeros((side, side), a.dtype); ya, xa = max(oy, 0), max(ox, 0); yb, xb = min(oy + side, 2048), min(ox + side, 2048)
    o[ya - oy:yb - oy, xa - ox:xb - ox] = a[ya:yb, xa:xb]; return o
rs = lambda a: cv2.resize(a, (R_, R_), interpolation=cv2.INTER_AREA)
dec = lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
S2 = np.zeros((len(HY), 4)); t0 = time.time()
for i, h in enumerate(HY):
    ch = CH[i]
    gts = [[poly_mask(p) > 0 for p, _ in view] for view in VREC[i]['gt']]
    if len(ch) == 0:
        for v in range(len(gts)): S2[i] += [0, 0, 0, len(gts[v])]
        continue
    labs = [dec(b) for b in h['labs']]; img = np.asarray(IMG[h['i']]); P0 = dec(h['prob'])
    nodes = []
    off = np.cumsum([0] + [int(l.max()) for l in labs])
    for c in ch:
        s_ = int(np.searchsorted(off, c, side='right') - 1); nodes.append(labs[s_] == (c - off[s_] + 1))
    allm = np.zeros((2048, 2048), np.uint8)
    for m in nodes: allm |= m.astype(np.uint8)
    xs, gs = [], []
    for m in nodes:
        oy, ox, side = geom(m); m8 = m.astype(np.uint8)
        xs.append(np.stack([rs(crop(img, oy, ox, side)), rs(crop(P0, oy, ox, side)), rs(crop(m8 * 255, oy, ox, side)), rs(crop((allm & (1 - m8)) * 255, oy, ox, side))])); gs.append((oy, ox, side))
    x = torch.from_numpy(np.stack(xs)).to(dev).float() / 255; acc = 0
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for rr in range(4):
            for fl in (0, 1):
                xx = torch.rot90(x, rr, (2, 3)); xx = xx.flip(3) if fl else xx
                p = models[int(folds[i])](xx).float().sigmoid(); p = p.flip(3) if fl else p; acc = acc + torch.rot90(p, -rr, (2, 3))
    PR = (acc[:, 0] / 8).cpu().numpy(); out = []
    for m, (oy, ox, side), pr in zip(nodes, gs, PR):
        p = cv2.resize(pr, (side, side), interpolation=cv2.INTER_LINEAR) >= 0.5
        full = np.zeros((2048, 2048), bool); ya, xa = max(oy, 0), max(ox, 0); yb, xb = min(oy + side, 2048), min(ox + side, 2048)
        full[ya:yb, xa:xb] = p[ya - oy:yb - oy, xa - ox:xb - ox]; out.append(full if full.any() else m)
    order = np.argsort([-m.sum() for m in nodes]); taken = np.zeros((2048, 2048), bool)
    for a_ in order: out[a_] = out[a_] & ~taken; taken |= out[a_]
    for v, gv in enumerate(gts):
        T = np.zeros((len(out), len(gv)))
        for a_, pm in enumerate(out):
            for b_, g in enumerate(gv):
                it_ = (pm & g).sum()
                if it_: T[a_, b_] = it_ / (pm | g).sum()
        best = {}
        for a_ in range(len(out)):
            for b_ in np.where(T[a_] > 0.5)[0]: best[b_] = max(best.get(b_, 0), T[a_, b_])
        tp = len(best); S2[i] += [sum(best.values()), tp, len(out) - tp, len(gv) - tp]
    if i % 100 == 0: print(f'stage2 {i}/707 {time.time() - t0:.0f}s', flush=True)
for nm, S_ in [('tree stage 1', S1), ('tree + coco refiner', S2)]:
    pf = [round(pq(S_[folds == k].sum(0)), 4) for k in range(5)]
    print(f'== {nm} (scorer trained on {MODE}): PQ {pq(S_.sum(0)):.4f} TP/FP/FN {S_.sum(0)[1:].astype(int)} folds {pf}', flush=True)
pickle.dump(dict(S1=S1, S2=S2, CH=CH, tau=tau), open(CGD + f'tree_pipeline_{MODE}.pkl', 'wb')); print('TREE DONE')
