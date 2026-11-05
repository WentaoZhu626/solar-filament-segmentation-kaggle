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


FPd = pd.read_csv(R + 'err/fp.csv')
def show_fp(i, title, fname):
    P, lab = kept_labels(i); n = lab.max()
    allg = [gt_masks(i, v) for v in range(len(views[OBS[i]]))]
    anyg = np.zeros((H, W), bool)
    for gl_ in allg:
        for g_ in gl_: anyg |= g_
    ov = np.bincount(lab[anyg], minlength=n + 1); area = np.bincount(lab.ravel(), minlength=n + 1)
    cand = [k for k in range(1, n + 1) if ov[k] == 0 and area[k] >= 600]
    if not cand: return False
    k = max(cand, key=lambda k: area[k]); m_ = lab == k
    ys, xs = np.nonzero(m_); mg = 90
    y0, y1, x0, x1 = max(ys.min() - mg, 0), min(ys.max() + mg, H), max(xs.min() - mg, 0), min(xs.max() + mg, W)
    side = max(y1 - y0, x1 - x0); y1 = min(y0 + side, H); x1 = min(x0 + side, W); c = np.s_[y0:y1, x0:x1]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
    ax[0].imshow(DATA[i][0][c], cmap='gray'); ax[0].set_title('image (limb-flattened)')
    ax[1].imshow(P[0][c] / 255, cmap='magma', vmin=0, vmax=1); ax[1].set_title('ensemble probability (mask)')
    ax[2].imshow(DATA[i][0][c], cmap='gray')
    for gl_ in allg:
        for g_ in gl_:
            if g_[c].any(): ax[2].contour(g_[c], [0.5], colors=['lime'], linewidths=1.5)
    for kk in np.unique(lab[c]):
        if kk and kk != k: ax[2].contour(lab[c] == kk, [0.5], colors=['cyan'], linewidths=1.0)
    ax[2].contour(m_[c], [0.5], colors=['red'], linewidths=2.0)
    ax[2].set_title('red = our false positive, cyan = our other filaments, green = any annotator', fontsize=9)
    for a in ax: a.axis('off')
    pin = P[0][m_].mean() / 255
    fig.suptitle(f'{title}\n{OBS[i]} | annotators on this image: {len(allg)} | FP area {area[k]} px, mean p {pin:.2f}', fontsize=11)
    plt.tight_layout(); plt.savefig(R + 'err/' + fname, dpi=80); plt.close()
    print(fname, OBS[i], len(allg), area[k], round(pin, 2), flush=True); return True
sp = FPd[(FPd.cat == 'spurious') & (~FPd.other_annotator_match)]
# (a) images with 3 annotators where NOBODY drew it, most confident first
a3 = sp[sp.n_views == 3].sort_values('score', ascending=False).drop_duplicates('i')
done = 0
for _, r in a3.iterrows():
    if show_fp(int(r.i), 'False positive: none of the 3 annotators drew anything here', f'fp3_{done}.png'):
        done += 1
        if done == 2: break
# (b) typical single-annotator FP (median score)
a1 = sp[(sp.n_views == 1) & (sp.area >= 600)].sort_values('score').drop_duplicates('i')
mid = a1.iloc[len(a1) // 2:]
done = 0
for _, r in mid.iterrows():
    if show_fp(int(r.i), 'Typical false positive (median filter score), single-annotator image', f'fp1_{done}.png'):
        done += 1
        if done == 2: break
print('FP VIZ DONE')
