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

# example 1: large filament split into pieces (single-annotator image, decent contrast)
F = G[(G.status == 'fragmented') & (G.area >= 3000) & (G.n_views == 1)].sort_values('contrast', ascending=False)
for _, r in F.head(3).iterrows():
    show(int(r.i), int(r.v), int(G[(G.i == r.i) & (G.v == r.v)].index.get_loc(_) if False else 0), '', 'tmp.png', '') if False else None
def gi_of(r):
    sub = G[(G.i == r.i) & (G.v == r.v)].reset_index()
    return int(np.where(sub['index'] == r.name)[0][0])
for n, (_, r) in enumerate(F.head(2).iterrows()):
    show(int(r.i), int(r.v), gi_of(r), 'Example 1: one annotated filament (green) split into several of our instances -> no piece reaches IoU 0.5 -> FN + FPs',
         f'ex1_{n}.png', f'contrast {r.contrast:.3f}, peak p {r.peak_p:.2f}')
# example 2: small filament, our instance too big (near miss)
S = G[(G.status == 'near miss') & (G.area < 600) & (G.n_views == 1)].sort_values('contrast', ascending=False)
done = 0
for _, r in S.head(40).iterrows():
    P, lab = kept_labels(int(r.i)); g = gt_masks(int(r.i), int(r.v))[gi_of(r)]
    k = np.bincount(lab[g], minlength=lab.max() + 1); k[0] = 0
    if not k.any(): continue
    a = (lab == k.argmax()).sum()
    if a / g.sum() > 1.8:
        show(int(r.i), int(r.v), gi_of(r), 'Example 2: small annotated filament (green), our instance (colour) drawn much larger -> IoU < 0.5 -> FN + FP',
             f'ex2_{done}.png', f'contrast {r.contrast:.3f}, peak p {r.peak_p:.2f}')
        done += 1
        if done == 2: break
print('VIZ DONE')
