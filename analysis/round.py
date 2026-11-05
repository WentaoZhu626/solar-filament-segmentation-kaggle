import sys, re
sys.argv = ['x', '--tag', 'err3', '--eval_only']
src = open('/root/autodl-tmp/run_ens.py').read()
def chunk(a, b):
    i, j = src.index(a), src.index(b)
    assert i < j; return src[i:j]
exec(src[:src.index('# ---------- 3) 5 folds')])
exec(chunk('# ---------- 7) evaluation', 'ALL = np.arange(len(OBS))'))
exec(chunk('# ===================== 8) post-processing B', '# ---------- 9) B on all OOF'))
exec(chunk('from scipy import ndimage as ndi', '# ---- 0) why are'))
import matplotlib.pyplot as plt
R = '/root/autodl-tmp/runs/err3/'
OOFM = np.load(R + 'oof_maps.npy', mmap_mode='r')
ST = pickle.load(open(R + 'err/state.pkl', 'rb')); BC = ST['BEST_C']; SC = ST['RESF'][BC]['SC']; S_THR = ST['RESF'][BC]['s']
G = pd.read_csv(R + 'err/gt.csv')

def kept_labels(i):
    P = np.array(OOFM[i]); sets, _, _ = candidate_sets(P, DATA[i][0])
    labB = sets['B']; aB = np.bincount(labB.ravel(), minlength=labB.max() + 1)[1:]
    idsB = np.where(aB >= 400)[0] + 1
    labE = sets[BC[1]]; idsE = np.arange(1, labE.max() + 1)
    keep = SC[i] >= S_THR; assert len(keep) == len(idsB) + len(idsE), (len(keep), len(idsB), len(idsE))
    out = np.zeros((H, W), np.int32); n = 0
    for j, k in enumerate(idsB):
        if keep[j]: n += 1; out[labB == k] = n
    for j, k in enumerate(idsE):
        if keep[len(idsB) + j]: n += 1; out[labE == k] = n
    return P, out

def gt_masks(i, v):
    return [poly_mask(p) > 0 for p, _ in views[OBS[i]][v]]

def show(i, v, gi, title, fname, note):
    P, lab = kept_labels(i); gts = gt_masks(i, v); g = gts[gi]
    ys, xs = np.nonzero(g); m = 60
    y0, y1, x0, x1 = max(ys.min() - m, 0), min(ys.max() + m, H), max(xs.min() - m, 0), min(xs.max() + m, W)
    side = max(y1 - y0, x1 - x0); y1 = min(y0 + side, H); x1 = min(x0 + side, W)
    c = np.s_[y0:y1, x0:x1]
    ids = [k for k in np.unique(lab[c]) if k]
    gl = []
    for k in ids:
        pk = lab == k; inter = (pk & g).sum(); u = (pk | g).sum()
        if inter: gl.append((k, inter / u, pk.sum()))
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
    ax[0].imshow(DATA[i][0][c], cmap='gray'); ax[0].set_title('image (limb-flattened)')
    ax[1].imshow(P[0][c] / 255, cmap='magma', vmin=0, vmax=1); ax[1].set_title('ensemble probability (mask)')
    ax[2].imshow(DATA[i][0][c], cmap='gray')
    for gg in gts:
        if gg[c].any(): ax[2].contour(gg[c], [0.5], colors=['lime'], linewidths=1.0, linestyles='dashed')
    ax[2].contour(g[c], [0.5], colors=['lime'], linewidths=2.2)
    cols = ['red', 'cyan', 'yellow', 'magenta', 'orange', 'deepskyblue', 'white']
    for n_, k in enumerate(ids):
        ax[2].contour((lab[c] == k), [0.5], colors=[cols[n_ % len(cols)]], linewidths=1.6)
    txt = ' | '.join(f'pred {n_+1}: IoU {iou:.2f}, {a} px' for n_, (k, iou, a) in enumerate(gl))
    ax[2].set_title('green = annotation (' + f'{g.sum()} px), colours = our filaments', fontsize=10)
    for a in ax: a.axis('off')
    fig.suptitle(f'{title}\n{OBS[i]}  view {v}  |  {txt}\n{note}', fontsize=11)
    plt.tight_layout(); plt.savefig(R + 'err/' + fname, dpi=80); plt.close()
    print(fname, OBS[i], 'gt px', g.sum(), gl, flush=True)


ALL = np.arange(len(OBS)); C = ST['C']
exec(chunk('# ---- 1) configurations', 'gbm = lambda'))
rows = []
for i in ALL:
    X, IOU, GID = pool(i, BC); kept = np.where(SC[i] >= S_THR)[0]
    for r_, j in enumerate(kept):
        x = X[j]
        rows.append(dict(i=i, rank=r_ + 1, area=float(np.exp(x[0])), elong=float(x[5]), inside=float(x[6]), contrast=float(x[7]),
                         around=float(x[6] + x[7]), mean_p=float(x[1]), score=float(SC[i][j]), n_views=len(GT_N[i]) if False else IOU.shape[1],
                         frac=float((IOU[j] > 0.5).mean()), anyv=bool((IOU[j] > 0.5).any()), best_iou=float(IOU[j].max())))
K = pd.DataFrame(rows); K.to_csv(R + 'err/kept.csv', index=False)
print('kept instances', len(K), '| features: elongation = sqrt(eig ratio); inside = preprocessed grey level (128 = background)')
print('inside (darkness) quantiles all kept:', K.inside.quantile([.01, .05, .25, .5, .75]).round(1).tolist())
K['shape'] = pd.cut(K.elong, [0, 1.5, 2, 3, 5, 1e9], labels=['round <1.5', '1.5-2', '2-3', '3-5', 'elongated >5'])
K['dark'] = pd.cut(K.inside, [-1, 20, 40, 60, 256], labels=['very dark <20', '20-40', '40-60', '>60'])
K['big'] = K.area >= 300
print('\n--- kept predictions (>= 300 px) by shape: count, share matched by >= 1 annotator, mean fraction of annotators matching ---')
print(K[K.big].groupby('shape', observed=False).agg(n=('frac', 'size'), any_match=('anyv', 'mean'), frac=('frac', 'mean'), inside=('inside', 'median')).round(3).to_string())
print('\n--- ROUND (elong < 2, >= 300 px) by darkness ---')
Rn = K[K.big & (K.elong < 2)]
print(Rn.groupby('dark', observed=False).agg(n=('frac', 'size'), any_match=('anyv', 'mean'), frac=('frac', 'mean'), around=('around', 'median'), area=('area', 'median')).round(3).to_string())
print('\n--- ROUND & dark (<40) on multi-annotator images: how many annotators drew them? ---')
m = Rn[(Rn.inside < 40) & (Rn.n_views > 1)]
print(pd.crosstab(m.n_views, (m.frac * m.n_views).round().astype(int), rownames=['annotators on image'], colnames=['# annotators whose filament matches (IoU>0.5)']).to_string())
# annotated filaments that are round-ish
G = pd.read_csv(R + 'err/gt.csv')
G['ratio'] = G.length / G.width.clip(lower=1)
print('\n--- ANNOTATED filaments by length/width ratio (disk ~1.3, line >> 3) ---')
G['shape'] = pd.cut(G.ratio, [0, 2, 3, 5, 1e9], labels=['round <2', '2-3', '3-5', 'elongated >5'])
print(G[G.area >= 300].groupby('shape', observed=False).agg(n=('area', 'size'), median_area=('area', 'median'), recall=('status', lambda s: (s == 'TP').mean())).round(3).to_string())
mg = G[(G.area >= 300) & (G.n_views > 1)]
print('agreement (share of OTHER annotators who drew the same filament) by shape, multi-annotator images:')
print(mg.groupby('shape', observed=False).agg(n=('agree', 'size'), mean_agree=('agree', 'mean'), nobody_else=('agree', lambda a: (a == 0).mean())).round(3).to_string())
# gallery of round & dark predictions
Gal = Rn[Rn.inside < 40].sort_values('inside').head(12)
fig, axs = plt.subplots(3, 4, figsize=(16, 12.5)); cache = {}
for ax, (_, r) in zip(axs.ravel(), Gal.iterrows()):
    i = int(r.i)
    if i not in cache: cache[i] = kept_labels(i)
    P, lab = cache[i]; m_ = lab == int(r['rank'])
    ys, xs = np.nonzero(m_); cy, cx = int(ys.mean()), int(xs.mean()); h = 110
    c = np.s_[max(cy - h, 0):cy + h, max(cx - h, 0):cx + h]
    ax.imshow(DATA[i][0][c], cmap='gray', vmin=0, vmax=255)
    for v in range(int(r.n_views)):
        for g_ in gt_masks(i, v):
            if g_[c].any(): ax.contour(g_[c], [0.5], colors=['lime'], linewidths=1.2)
    ax.contour(m_[c], [0.5], colors=['red'], linewidths=1.5)
    ax.set_title(f'{OBS[i][:14]} {OBS[i][-7:-5]} | matched {int(round(r.frac * r.n_views))}/{int(r.n_views)} annot.\narea {int(r.area)} px, elong {r.elong:.2f}, grey {r.inside:.0f}', fontsize=9)
    ax.axis('off')
fig.suptitle('Round & very dark predictions (red) vs annotations (green); grey 128 = background', fontsize=12)
plt.tight_layout(); plt.savefig(R + 'err/round_gallery.png', dpi=70); plt.close()
print('ROUND DONE', flush=True)
