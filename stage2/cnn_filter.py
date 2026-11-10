
"""CNN judge for ens3 candidates (replaces / augments the 13-feature GBM FP filter).
stage crops : 7-channel 128 px crops per candidate (local: img, prob, mask; context 512 px: img, prob, mask, other candidates)
stage train : 5-fold ResNet, soft BCE on y = share of annotators matching the candidate (IoU > 0.5) -> OOF cnn score
stage eval  : PQ of keep rules: current filter / cnn / GBM(13 feats) / GBM(13 feats + cnn) / blends"""
import os, sys, time, pickle, numpy as np, cv2
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
sys.path.insert(0, '/root/autodl-tmp/analysis')
B = '/root/autodl-tmp/'; OUT = os.environ.get('CNNF_OUT', B + 'runs/cnnf/'); VIS = os.environ.get('VIS', B + 'runs/vis/vis.pkl'); os.makedirs(OUT, exist_ok=True)
STAGE = sys.argv[1]; S = int(os.environ.get('S', 128)); TAG = os.environ.get('TAG', ''); CROPS = (OUT + 'crops.npy') if S == 128 else f'/root/cnnf_cache/crops{S}.npy'; os.makedirs('/root/cnnf_cache', exist_ok=True)
def load_rec(): return pickle.load(open(VIS, 'rb'))
def crops_for(r, img):
    import err_viz as ev
    prob = cv2.imdecode(np.frombuffer(r['prob_png'], np.uint8), cv2.IMREAD_UNCHANGED)
    inst = ev.instances(r, kept_only=False); allm = np.zeros((2048, 2048), np.uint8)
    for m in inst: allm |= m.astype(np.uint8)
    out = []
    for m in inst:
        ys, xs = np.nonzero(m); cy, cx = ys.mean(), xs.mean()
        side = max(ys.max() - ys.min(), xs.max() - xs.min()) * 1.5 + 32; side = max(side, 64)
        def cut(a, sd, fill, interp):
            h = int(sd // 2); x0, y0 = int(cx) - h, int(cy) - h; pad = h + 2
            ap = cv2.copyMakeBorder(a, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=fill)
            c = ap[y0 + pad:y0 + pad + 2 * h, x0 + pad:x0 + pad + 2 * h]
            return cv2.resize(c, (S, S), interpolation=interp)
        mm = m.astype(np.uint8) * 255; oth = ((allm > 0) & ~m).astype(np.uint8) * 255
        ch = [cut(img, side, 128, cv2.INTER_AREA), cut(prob, side, 0, cv2.INTER_AREA), cut(mm, side, 0, cv2.INTER_AREA),
              cut(img, 512, 128, cv2.INTER_AREA), cut(prob, 512, 0, cv2.INTER_AREA), cut(mm, 512, 0, cv2.INTER_AREA), cut(oth, 512, 0, cv2.INTER_AREA)]
        out.append(np.stack(ch))
    return np.stack(out) if out else np.zeros((0, 7, S, S), np.uint8)
if STAGE == 'crops':
    from joblib import Parallel, delayed
    V = load_rec(); R = V['rec']; IMG = np.load(B + 'cache/img.npy', mmap_mode='r'); t0 = time.time()
    def one(k): r = R[k]; return crops_for(r, np.asarray(IMG[r['i']]))
    Xs = Parallel(n_jobs=14, batch_size=4)(delayed(one)(k) for k in range(len(R)))
    X = np.concatenate(Xs); meta = []
    for k, r in enumerate(R):
        for j in range(len(r['kept'])):
            io = r['iou'][j].astype(np.float32)
            meta.append(dict(k=k, j=j, fold=r['fold'], y=float((io > 0.5).mean()), ious=io, score=float(r['score'][j]), kept=bool(r['kept'][j]), F=r['X'][j]))
    assert len(meta) == len(X), (len(meta), len(X))
    np.save(CROPS, X); pickle.dump(meta, open(OUT + 'meta.pkl', 'wb'))
    print('crops', X.shape, f'{time.time() - t0:.0f}s', flush=True)
if STAGE == 'train':
    import torch, torch.nn as nn, timm
    X = np.load(CROPS, mmap_mode='r'); meta = pickle.load(open(OUT + 'meta.pkl', 'rb'))
    y = np.array([m['y'] for m in meta], np.float32); fold = np.array([m['fold'] for m in meta])
    ARCH = os.environ.get('ARCH', 'resnet34'); EP = int(os.environ.get('EP', 15)); DEV = 'cuda'
    Xt = torch.from_numpy(np.ascontiguousarray(X)).to(DEV)          # uint8 on GPU (~1 GB)
    yt = torch.from_numpy(y).to(DEV); oof = np.zeros(len(y), np.float32); t0 = time.time()
    def aug(x):
        if np.random.rand() < 0.5: x = x.flip(-1)
        if np.random.rand() < 0.5: x = x.flip(-2)
        return torch.rot90(x, np.random.randint(4), (2, 3))
    for k in range(5):
        tr = np.where(fold != k)[0]; va = np.where(fold == k)[0]
        model = timm.create_model(ARCH, pretrained=True, in_chans=7, num_classes=1, drop_rate=0.2).to(DEV)
        LR = float(os.environ.get('LR', 3e-4)); opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2); steps = EP * (len(tr) // 128 + 1)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
        for ep in range(EP):
            model.train(); perm = np.random.permutation(tr); tl = 0
            BS = int(os.environ.get('BS', 128))
            for b in range(0, len(perm), BS):
                idx = torch.from_numpy(perm[b:b + BS]).to(DEV); x = aug(Xt[idx].float() / 255)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = nn.functional.binary_cross_entropy_with_logits(model(x)[:, 0].float(), yt[idx])
                opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tl += loss.item() * len(idx)
        model.eval(); p = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for b in range(0, len(va), 256):
                x = Xt[torch.from_numpy(va[b:b + 256]).to(DEV)].float() / 255; acc = 0
                for q in range(4): acc = acc + torch.sigmoid(model(torch.rot90(x, q, (2, 3)))[:, 0].float())
                p.append((acc / 4).cpu().numpy())
        oof[va] = np.concatenate(p); torch.save(model.state_dict(), OUT + f'cnn{TAG}_fold{k}.pth')
        from sklearn.metrics import roc_auc_score
        print(f'fold {k} train loss {tl / len(tr):.4f} | val AUC(y>0.5) {roc_auc_score(y[va] > 0.5, oof[va]):.4f} | {time.time() - t0:.0f}s', flush=True)
    np.save(OUT + f'oof{TAG}.npy', oof)
if STAGE == 'eval':
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import roc_auc_score
    meta = pickle.load(open(OUT + 'meta.pkl', 'rb')); V = load_rec(); R = V['rec']
    NGT = sum(len(v) for r in R for v in r['gt_match'])
    IO = [m['ious'] for m in meta]; y = np.array([m['y'] for m in meta]); fold = np.array([m['fold'] for m in meta])
    SI = np.array([x[x > 0.5].sum() for x in IO]); TPc = np.array([(x > 0.5).sum() for x in IO]); NVc = np.array([len(x) for x in IO])
    def pq(keep, sel=None):
        sel = np.ones(len(keep), bool) if sel is None else sel; keep = keep & sel
        ng = NGT if sel.all() else sum(len(v) for r in R if r['fold'] == fold[sel][0] for v in r['gt_match'])
        s, tp, n = SI[keep].sum(), TPc[keep].sum(), NVc[keep].sum()
        return s / (tp + 0.5 * (n - tp) + 0.5 * (ng - tp)), int(tp), int(n - tp), int(ng - tp)
    score = np.array([m['score'] for m in meta]); kept = np.array([m['kept'] for m in meta]); F = np.stack([m['F'] for m in meta])
    print('check current kept:', pq(kept))
    cnn = np.load(OUT + f'oof{TAG}.npy')
    def gbm(Z):
        p = np.zeros(len(y))
        for k in range(5):
            tr, va = fold != k, fold == k
            p[va] = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(Z[tr], y[tr]).predict(Z[va])
        return p
    def sweep(sc, name):
        res = {s: pq(sc >= s) for s in np.round(np.arange(0.15, 0.66, 0.025), 3)}
        b = max(res, key=lambda s: res[s][0])
        # nested: threshold chosen on the other 4 folds
        nested = []
        for k in range(5):
            selk = fold == k; tr = ~selk
            def pq_sub(keep, m):
                ks = set(np.where(m)[0]); folds_m = set(fold[m]); ng = sum(len(v) for r in R if r['fold'] in folds_m for v in r['gt_match'])
                kk = keep & m; s_, tp, n = SI[kk].sum(), TPc[kk].sum(), NVc[kk].sum(); return s_, tp, n - tp, ng - tp
            sb = max(res, key=lambda s: (lambda a: a[0] / (a[1] + 0.5 * a[2] + 0.5 * a[3]))(pq_sub(sc >= s, tr)))
            nested.append(pq_sub(sc >= sb, selk))
        tot = np.sum(nested, 0); pn = tot[0] / (tot[1] + 0.5 * tot[2] + 0.5 * tot[3])
        print(f'{name:30s} AUC(y>0.5) {roc_auc_score(y > 0.5, sc):.4f} | best s {b}: PQ {res[b][0]:.4f} TP/FP/FN {res[b][1:]} | nested-threshold PQ {pn:.4f}', flush=True)
    sweep(score, 'current filter (13 feats)')
    sweep(cnn, 'CNN alone')
    g13 = gbm(F); sweep(g13, 'GBM 13 feats (refit)')
    sweep(gbm(np.c_[F, cnn]), 'GBM 13 feats + CNN')
    sweep(gbm(np.c_[score, cnn]), 'GBM(score, CNN)')
    for w in (0.3, 0.5, 0.7): sweep(w * cnn + (1 - w) * score, f'blend {w:.1f} CNN + {1 - w:.1f} score')
