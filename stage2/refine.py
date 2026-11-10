
# Boundary refiner: re-segment each candidate in a local crop (image, ensemble prob, own mask, other candidates)
# target = mean of the annotator GT masks that overlap the candidate (IoU > 0.2). Out-of-fold, then official-style PQ.
import os, sys, time, pickle, numpy as np, cv2
from joblib import Parallel, delayed
STAGE = sys.argv[1]; R = 256
VIS = os.environ.get('VIS', '/root/autodl-tmp/runs/vis/vis.pkl'); OUT = os.environ.get('RF_OUT', '/root/refine/'); os.makedirs(OUT, exist_ok=True)
V = pickle.load(open(VIS, 'rb')); REC = V['rec']
IMG = np.load('/root/autodl-tmp/cache/img.npy', mmap_mode='r')
def dec(b): return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
def polymask(poly):
    m = np.zeros((2048, 2048), np.uint8); cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1); return m
def cand_masks(r):
    labs = {'B': dec(r['B_png']), 0.3: dec(r['F_png'])}
    return [labs[s] == l for s, l in r['cand_src']]
def geom(m):
    ys, xs = np.where(m); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(max(y1 - y0, x1 - x0) * 1.3 + 48); cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    return cy - side // 2, cx - side // 2, side
def crop(a, oy, ox, side):
    out = np.zeros((side, side), a.dtype)
    ya, xa = max(oy, 0), max(ox, 0); yb, xb = min(oy + side, 2048), min(ox + side, 2048)
    out[ya - oy:yb - oy, xa - ox:xb - ox] = a[ya:yb, xa:xb]; return out
def rs(a, interp): return cv2.resize(a, (R, R), interpolation=interp)

if STAGE == 'crops':
    def one(n, r, img):
        i = r['i']; P = dec(r['prob_png']); M = cand_masks(r)
        allm = np.zeros((2048, 2048), np.uint8)
        for m in M: allm |= m.astype(np.uint8)
        GTM = {}
        X, Y, meta = [], [], []
        for j, m in enumerate(M):
            oy, ox, side = geom(m); mm = m.astype(np.uint8)
            other = (allm & (1 - mm))
            x = np.stack([rs(crop(img, oy, ox, side), cv2.INTER_AREA), rs(crop(P, oy, ox, side), cv2.INTER_AREA),
                          rs(crop(mm * 255, oy, ox, side), cv2.INTER_AREA), rs(crop(other * 255, oy, ox, side), cv2.INTER_AREA)])
            ts = []
            for v in range(r['n_views']):
                if r['iou'][j, v] > 0.2:
                    g = int(r['gid'][j, v]); key = (v, g)
                    if key not in GTM: GTM[key] = polymask(r['gt'][v][g][0])
                    ts.append(rs(crop(GTM[key] * 255, oy, ox, side), cv2.INTER_AREA).astype(np.float32))
            y = np.mean(ts, 0).astype(np.uint8) if ts else np.zeros((R, R), np.uint8)
            X.append(x); Y.append(y); meta.append((n, j, r['fold'], bool(r['kept'][j]), oy, ox, side, len(ts)))
        return X, Y, meta
    t0 = time.time()
    res = Parallel(n_jobs=int(os.environ.get('NJ', 12)))(delayed(one)(n, REC[n], np.array(IMG[REC[n]['i']])) for n in range(len(REC)))
    X = np.stack([x for a in res for x in a[0]]); Y = np.stack([y for a in res for y in a[1]])
    META = np.array([m for a in res for m in a[2]], dtype=object)
    np.save(OUT + 'X.npy', X); np.save(OUT + 'Y.npy', Y); pickle.dump(META, open(OUT + 'meta.pkl', 'wb'))
    print('crops', X.shape, 'with target', sum(m[7] > 0 for m in META), f'{time.time() - t0:.0f}s')

if STAGE == 'train':
    import torch, torch.nn.functional as F, segmentation_models_pytorch as smp
    torch.backends.cudnn.benchmark = True
    X = np.load(OUT + 'X.npy', mmap_mode='r'); Y = np.load(OUT + 'Y.npy'); META = pickle.load(open(OUT + 'meta.pkl', 'rb'))
    X = np.array(X)
    fold = np.array([m[2] for m in META]); has = np.array([m[7] > 0 for m in META])
    EP = int(os.environ.get('EP', 20)); BS = 32; dev = 'cuda'
    PRED = np.zeros((len(X), R, R), np.uint8)
    for k in range(5):
        torch.manual_seed(k)
        net = smp.Unet('resnet34', encoder_weights='imagenet', in_channels=4, classes=1).to(dev).to(memory_format=torch.channels_last)
        tr = np.where((fold != k) & has)[0]; te = np.where(fold == k)[0]
        steps = EP * (len(tr) // BS)
        opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4 * 3, total_steps=steps, pct_start=0.1)
        scaler = torch.cuda.amp.GradScaler(); t0 = time.time(); st = 0
        for ep in range(EP):
            net.train(); perm = np.random.RandomState(ep + 10 * k).permutation(tr); tl = 0
            for b in range(len(tr) // BS):
                idx = np.sort(perm[b * BS:(b + 1) * BS])
                x = torch.from_numpy(X[idx]).to(dev).float() / 255; y = torch.from_numpy(Y[idx]).to(dev).float()[:, None] / 255
                rr = np.random.randint(4); x = torch.rot90(x, rr, (2, 3)); y = torch.rot90(y, rr, (2, 3))
                if np.random.rand() < 0.5: x = x.flip(3); y = y.flip(3)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    lo = net(x.contiguous(memory_format=torch.channels_last)).float()
                p = lo.sigmoid(); inter = (p * y).sum((1, 2, 3)); dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
                loss = F.binary_cross_entropy_with_logits(lo, y) + dice.mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step(); tl += loss.item()
            print(f'fold {k} ep {ep} loss {tl / (len(tr) // BS):.4f} {time.time() - t0:.0f}s', flush=True)
        net.eval()
        with torch.no_grad():
            for b in range(0, len(te), 64):
                idx = te[b:b + 64]; x = torch.from_numpy(X[idx]).to(dev).float() / 255; acc = 0
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    for rr in range(4):
                        for fl in (0, 1):
                            xx = torch.rot90(x, rr, (2, 3)); xx = xx.flip(3) if fl else xx
                            p = net(xx.contiguous(memory_format=torch.channels_last)).float().sigmoid()
                            p = p.flip(3) if fl else p; acc = acc + torch.rot90(p, -rr, (2, 3))
                PRED[idx] = (acc[:, 0] / 8 * 255).round().byte().cpu().numpy()
        torch.save(net.state_dict(), OUT + f'ref_fold{k}.pth'); del net; torch.cuda.empty_cache()
    np.save(OUT + 'pred.npy', PRED); print('TRAIN DONE')

if STAGE == 'eval':
    PRED = np.load(OUT + 'pred.npy', mmap_mode='r'); META = pickle.load(open(OUT + 'meta.pkl', 'rb'))
    if os.environ.get('KEEP') == 'cnn':
        CO = '/root/autodl-tmp/runs/cnnf/'
        cn = np.mean([np.load(CO + f'oof{t}.npy') for t in ['', '_cnxt', '_effb3', '_r50', '_r34_192']], 0)
        gb = np.concatenate([r['score'] for r in REC]); assert len(cn) == len(gb) == len(META), (len(cn), len(gb), len(META))
        fin = 0.6 * cn + 0.4 * gb
        META = np.array([tuple(m[:3]) + (bool(fin[c] >= 0.4),) + tuple(m[4:]) for c, m in enumerate(META)], dtype=object)
        print('cnn keep', sum(m[3] for m in META), flush=True)
    byimg = {}
    for c, m in enumerate(META): byimg.setdefault(m[0], []).append(c)
    THS = [float(t) for t in os.environ.get('THS', '0.3,0.4,0.5,0.6').split(',')]
    def iou_tab(preds, gts):
        T = np.zeros((len(preds), len(gts)))
        for a, p in enumerate(preds):
            if p is None: continue
            for b, g in enumerate(gts):
                it = (p & g).sum()
                if it: T[a, b] = it / ((p | g).sum())
        return T
    def score(T):
        best = {}
        for a in range(T.shape[0]):
            for b in np.where(T[a] > 0.5)[0]:
                if T[a, b] > best.get(b, (0, -1))[0]: best[b] = (T[a, b], a)
        return np.array([sum(x[0] for x in best.values()), len(best), T.shape[0] - len(best), T.shape[1] - len(best)])
    def one(n, r, cs, metas, preds):
        M = cand_masks(r); META_ = dict(zip(cs, metas)); PR_ = dict(zip(cs, preds)); kept = [c for c in cs if META_[c][3]]
        gts = [[polymask(p) > 0 for p, _ in r['gt'][v]] for v in range(r['n_views'])]
        variants = {'orig': [M[META_[c][1]] for c in kept]}
        for t in THS:
            out = []
            for c in kept:
                _, j, _, _, oy, ox, side, _ = META_[c]
                p = cv2.resize(PR_[c], (side, side), interpolation=cv2.INTER_LINEAR) >= t * 255
                full = np.zeros((2048, 2048), bool)
                ya, xa = max(oy, 0), max(ox, 0); yb, xb = min(oy + side, 2048), min(ox + side, 2048)
                full[ya:yb, xa:xb] = p[ya - oy:yb - oy, xa - ox:xb - ox]
                if not full.any(): full = M[j]
                out.append(full)
            # resolve overlaps: earlier (larger orig area first) keeps pixel
            order = np.argsort([-M[META_[c][1]].sum() for c in kept]); taken = np.zeros((2048, 2048), bool)
            for a in order: out[a] = out[a] & ~taken; taken |= out[a]
            variants[t] = out
        res = {}
        for name, preds in variants.items():
            S = np.zeros(4); per = []
            for v in range(r['n_views']):
                T = iou_tab(preds, gts[v]); s = score(T); S += s
            res[name] = S
        return res
    t0 = time.time()
    res = Parallel(n_jobs=int(os.environ.get('NJ', 12)))(delayed(one)(n, REC[n], byimg[n], [META[c] for c in byimg[n]], [np.array(PRED[c]) for c in byimg[n]]) for n in range(len(REC)))
    folds = np.array([r['fold'] for r in REC])
    for name in res[0]:
        S = sum(x[name] for x in res); pq = S[0] / (S[1] + 0.5 * S[2] + 0.5 * S[3])
        pf = []
        for k in range(5):
            Sk = sum(res[n][name] for n in range(len(REC)) if folds[n] == k); pf.append(round(Sk[0] / (Sk[1] + 0.5 * Sk[2] + 0.5 * Sk[3]), 4))
        print(name, f'PQ {pq:.4f} SQ {S[0] / max(S[1], 1):.4f} TP/FP/FN {S[1:].astype(int)}', pf, flush=True)
    print(f'EVAL DONE {time.time() - t0:.0f}s')
