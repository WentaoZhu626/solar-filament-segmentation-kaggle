# ===== Solar Filament Segmentation 2026 — FINAL PIPELINE, INFERENCE ONLY (no training) =====
# Stage 1 (as in infer_ens.py): limb-darkening flattening -> mean of 15 U-Nets (3 encoders x 5 folds, mask + spine,
#   hflip TTA) -> spine-merged instances (>= 400 px) + faint candidates (p > 0.3).
# Stage 2: keep rule 0.6 * mean(5 CNN judges x 5 folds, 4-rotation TTA) + 0.4 * GBM filter >= 0.4, then the boundary
#   refiner (U-Net on a crop around each kept filament, 5 folds x D4 TTA, threshold 0.5); pixel conflicts go to the
#   larger instance, so the output never overlaps. Out-of-fold PQ 0.4541 (official evaluation), public LB 0.40.
# All weights are read from ../weights (stored in this repository). The GBM filter is refit here (a few seconds) on the
# exported out-of-fold candidate table weights/stage2_filter_table.npz.
# usage: python inference/predict_final.py [--test_dir DIR] [--weights DIR] [--out FILE]
import os, sys, glob, json, time, argparse
import cv2, numpy as np, pandas as pd, torch, torch.nn.functional as F
import segmentation_models_pytorch as smp
from pycocotools import mask as mu
from skimage.morphology import skeletonize
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--test_dir', default=None, help='folder with the test JPEGs (default: search /kaggle/input and ./data)')
ap.add_argument('--weights', default=os.path.join(HERE, '..', 'weights'))
ap.add_argument('--out', default='submission.csv')
ARGS, _ = ap.parse_known_args()
H = W = 2048; DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
if ARGS.test_dir:
    TEST_DIR = ARGS.test_dir.rstrip('/') + '/'
else:
    TEST_DIR = (glob.glob('/kaggle/input/**/test/test_images', recursive=True) + glob.glob('./data/**/test_images', recursive=True))[0] + '/'
WDIR = os.path.abspath(ARGS.weights) + '/'
EXP = json.load(open(WDIR + 'filter_export.json'))
MEMBERS = [m.split(':') for m in EXP['encoder'].split(',')]          # [[timm encoder, weight folder], ...]
CFG = dict(encoder=MEMBERS[0][0], spine_loss_w=0.5, cldice_w=0.2, cldice_iters=8, n_folds=5)
print('test images:', len(os.listdir(TEST_DIR)), '| weights:', WDIR, '| members:', MEMBERS, '| device:', DEV)
def soft_dice(p, t, eps=1.0):
    return 1 - ((2 * (p * t).sum((2, 3)) + eps) / (p.sum((2, 3)) + t.sum((2, 3)) + eps)).mean()


MODEL_CODE = r'''
def soft_dice(p, t, eps=1.0):
    return 1 - ((2 * (p * t).sum((2, 3)) + eps) / (p.sum((2, 3)) + t.sum((2, 3)) + eps)).mean()

def _gauss(s):
    r = int(3 * s); x = torch.arange(-r, r + 1, dtype=torch.float32); k = torch.exp(-x ** 2 / (2 * s * s))
    return k / k.sum()

class MultiView(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer('k_small', _gauss(1.5)); self.register_buffer('k_large', _gauss(12.0))

    def blur(self, x, k):
        r = len(k) // 2
        x = F.conv2d(F.pad(x, (r, r, 0, 0), mode='reflect'), k.view(1, 1, 1, -1))
        return F.conv2d(F.pad(x, (0, 0, r, r), mode='reflect'), k.view(1, 1, -1, 1))

    def forward(self, x):
        with torch.autocast('cuda', enabled=False):
            x = x.float()
            mu = self.blur(x, self.k_large)
            sd = self.blur((x - mu) ** 2, self.k_large).clamp_min(1e-6).sqrt()
            lcn = ((x - mu) / (sd + 0.02)).clamp(-4, 4) * 0.12 + 0.5
            x3 = torch.cat([x, self.blur(x, self.k_small), lcn], 1)
        return self.net(x3)

def make_model(pretrained=True):
    return MultiView(smp.Unet(CFG['encoder'], encoder_weights='imagenet' if pretrained else None, in_channels=3, classes=2)).to(DEV)

def _erode(x):
    return -F.max_pool2d(-x, 3, 1, 1)

def _open(x):
    return F.max_pool2d(_erode(x), 3, 1, 1)

def soft_skel(x, it):
    skel = F.relu(x - _open(x))
    for _ in range(it):
        x = _erode(x)
        d = F.relu(x - _open(x))
        skel = skel + F.relu(d - skel * d)
    return skel

def cl_dice(p, t, it):
    sp, st = soft_skel(p, it), soft_skel(t, it)
    tprec = ((sp * t).sum((2, 3)) + 1) / (sp.sum((2, 3)) + 1)
    tsens = ((st * p).sum((2, 3)) + 1) / (st.sum((2, 3)) + 1)
    return (1 - 2 * tprec * tsens / (tprec + tsens)).mean()

def loss_fn(logits, t):
    """mask: BCE + soft Dice + clDice (aux); spine: BCE + soft Dice, weighted by spine_loss_w."""
    pm = logits[:, :1].sigmoid()
    lm = (F.binary_cross_entropy_with_logits(logits[:, :1], t[:, :1]) + soft_dice(pm, t[:, :1])
          + CFG['cldice_w'] * cl_dice(pm, t[:, :1], CFG['cldice_iters']))
    ls = F.binary_cross_entropy_with_logits(logits[:, 1:], t[:, 1:]) + soft_dice(logits[:, 1:].sigmoid(), t[:, 1:])
    return lm + CFG['spine_loss_w'] * ls, lm.item(), ls.item()

'''
exec(MODEL_CODE)


# ---------- 1) preprocessing: flatten limb darkening so filaments look equally dark at disk centre and near the limb ----------
def disk_mask(img):
    """Solar disk = largest blob above the Otsu threshold."""
    _, m = cv2.threshold(cv2.GaussianBlur(img, (0, 0), 5), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    return (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.float32)

def preprocess(path):
    """-> uint8 image: disk background ~128, dark filaments toward 0, off-disk = 128."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    d = disk_mask(img)
    f = img.astype(np.float32)
    small = lambda a: cv2.resize(a, (W // 8, H // 8), interpolation=cv2.INTER_AREA)    # smooth background at 1/8 size (fast)
    bg = cv2.GaussianBlur(small(f * d), (0, 0), 4) / (cv2.GaussianBlur(small(d), (0, 0), 4) + 1e-6)   # disk pixels only
    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR)
    flat = np.where(d > 0, f / (bg + 1e-6), 1.0)
    return np.clip((flat - 1.0) * 510 + 128, 0, 255).astype(np.uint8)    # +/-25% contrast -> full 8-bit range



@torch.no_grad()
def predict(model, img_u8):
    """2048x2048 inference with horizontal-flip TTA -> float32 [2, H, W] = (mask prob, spine prob)."""
    model.eval()
    x = torch.from_numpy(img_u8.astype(np.float32) / 255)[None, None].to(DEV)
    with torch.autocast('cuda', dtype=torch.float16):
        p = model(x).sigmoid() + model(x.flip(-1)).sigmoid().flip(-1)
    return (p[0] / 2).float().cpu().numpy()



def foreground(prob_u8, thr, lo=None):
    """Binary filament mask. lo=None: plain threshold. Otherwise HYSTERESIS: pixels > lo that are 8-connected to at least
    one pixel > thr -- thr decides WHERE a filament is (no new noise blobs), lo decides HOW WIDE it is drawn."""
    if lo is None:
        return (prob_u8 > thr * 255).astype(np.uint8)
    n, lab = cv2.connectedComponents((prob_u8 > lo * 255).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, np.uint8); keep[np.unique(lab[prob_u8 > thr * 255])] = 1; keep[0] = 0
    return keep[lab]

def to_instances(prob_u8, thr, min_area, lo=None):
    """Post-processing A: (hysteresis) threshold -> 8-connected components -> drop small blobs. Returns int32 label map."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(foreground(prob_u8, thr, lo), connectivity=8)
    keep = np.zeros(n, np.int32)
    big = np.where(st[1:, cv2.CC_STAT_AREA] >= min_area)[0] + 1
    keep[big] = np.arange(1, len(big) + 1)
    return keep[lab]



# ===================== 8) post-processing B: re-join fragments of one filament using the predicted spine =====================
# Starts from A's blobs and NEVER splits them; it only MERGES blobs that belong to the same spine:
#   1) skeletonise the spine map -> thin centre lines (tiny pieces < min_spine px are dropped)
#   2) link spine pieces whose end points are within `link` px AND point toward each other (angle < 45 deg)
#   3) blobs touched by the same (linked) spine become one instance -> a filament broken by a faint gap stays ONE filament.
# (A split-and-merge variant -- watershed seeded by spines -- was tested on fold 0 and was worse than A: it over-split
#  filaments whose spine had several pieces. PQ fold 0: A 0.404, split-B 0.385-0.399, merge-only B 0.410.)
NB8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.float32)

def find(par, a):
    while par[a] != a:
        par[a] = par[par[a]]; a = par[a]
    return a

def link_spines(sk_lab, n, max_dist, max_angle=45, r=10):
    """Union-find over skeleton pieces 1..n-1: merge two pieces if an end point of each lies within max_dist
    and both end directions point toward each other (angle < max_angle)."""
    par = list(range(n))
    if max_dist <= 0 or n < 3:
        return par
    sk = (sk_lab > 0).astype(np.float32)
    nb = cv2.filter2D(sk, -1, NB8, borderType=cv2.BORDER_CONSTANT)
    ey, ex = np.nonzero((sk > 0) & (nb == 1))                 # end points have exactly one skeleton neighbour
    py, px = np.nonzero(sk); pl = sk_lab[py, px]
    E = []
    for y, x in zip(ey, ex):
        k = sk_lab[y, x]
        near = (pl == k) & (np.abs(py - y) <= r) & (np.abs(px - x) <= r)
        d = np.array([x - px[near].mean(), y - py[near].mean()])   # outward direction of this end
        E.append((k, np.array([x, y], float), d / (np.linalg.norm(d) + 1e-6)))
    c = np.cos(np.deg2rad(max_angle))
    for a in range(len(E)):
        for b in range(a + 1, len(E)):
            (ka, pa, da), (kb, pb, db) = E[a], E[b]
            if ka == kb:
                continue
            v = pb - pa; dist = np.linalg.norm(v)
            if 0 < dist <= max_dist and da @ (v / dist) > c and db @ (-v / dist) > c:
                par[find(par, ka)] = find(par, kb)
    return par

def spine_merge_instances(pm, ps, thr, min_area, link=40, lo=None, thr_s=0.5, min_spine=10):
    """Post-processing B. pm, ps: uint8 mask / spine probability maps. Returns a non-overlapping int32 label map."""
    fg = foreground(pm, thr, lo)
    ncc, cc = cv2.connectedComponents(fg, connectivity=8)          # A's blobs (before the size filter)
    par_cc = list(range(ncc))
    grp = cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * link + 1, 2 * link + 1))) if link > 0 else fg
    n, glab, st, _ = cv2.connectedComponentsWithStats(grp, connectivity=8)   # independent neighbourhoods -> work on crops
    for g in range(1, n):
        x, y, w, h = st[g, :4]
        sl = np.s_[y:y + h, x:x + w]
        region = glab[sl] == g
        ccs = cc[sl][region]
        if len(np.unique(ccs[ccs > 0])) < 2:                       # nothing to merge in this neighbourhood
            continue
        sk = skeletonize((ps[sl] > thr_s * 255) & region)
        k, sk_lab, sst, _ = cv2.connectedComponentsWithStats(sk.astype(np.uint8), connectivity=8)
        small = np.where(sst[1:, cv2.CC_STAT_AREA] < min_spine)[0] + 1
        sk_lab[np.isin(sk_lab, small)] = 0
        par = link_spines(sk_lab, k, link)
        roots = np.array([find(par, j) for j in range(k)])
        on = (sk_lab > 0) & (cc[sl] > 0)
        pairs = np.unique(np.stack([roots[sk_lab[on]], cc[sl][on]], 1), axis=0)   # (spine root, blob it touches)
        for r in np.unique(pairs[:, 0]):
            blobs = pairs[pairs[:, 0] == r, 1]
            for b in blobs[1:]:
                par_cc[find(par_cc, b)] = find(par_cc, blobs[0])
    root = np.array([find(par_cc, j) for j in range(ncc)])
    area = np.bincount(root[cc].ravel(), minlength=ncc); area[0] = 0     # size filter AFTER merging
    keep = np.zeros(ncc, np.int32)
    big = np.where(area >= min_area)[0]
    keep[big] = np.arange(1, len(big) + 1)
    return keep[root[cc]]

def instances(method, cfg, P):
    """P = uint8 [2, H, W] (mask prob, spine prob) -> label map with post-processing A or B."""
    if method == 'B':
        return spine_merge_instances(P[0], P[1], **cfg)
    return to_instances(P[0], cfg['thr'], cfg['min_area'], cfg.get('lo'))


from sklearn.ensemble import HistGradientBoostingRegressor
from scipy import ndimage as ndi
B_CFG = dict(thr=0.6, min_area=50, lo=None, link=40)         # B instances down to 50 px; size cut applied per configuration
T_LOW, MIN_EXTRA = [0.5, 0.4, 0.3], 50
FEATS = ['log area', 'mean p', 'max p', 'frac p>0.8', 'mean spine p', 'elongation', 'img inside', 'contrast', 'radius',
         'ring p', 'dist to big filament', 'log #big filaments', 'source t']
K15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)); K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def cand_features(lab, P, img, dist_big, n_big, src_t):
    """Per-candidate features computed on small crops (fast). lab: int32 label map of this candidate set."""
    pm, ps = P[0].astype(np.float32) / 255, P[1].astype(np.float32) / 255
    rows = []
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            rows.append(np.zeros(len(FEATS), np.float32)); continue
        y0, y1 = max(sl[0].start - 8, 0), min(sl[0].stop + 8, H); x0, x1 = max(sl[1].start - 8, 0), min(sl[1].stop + 8, W)
        c = np.s_[y0:y1, x0:x1]; m = lab[c] == k
        ring = (cv2.dilate(m.astype(np.uint8), K15) > 0) & (lab[c] == 0)
        ys, xs = np.nonzero(m); a = len(ys)
        cov = np.cov(np.stack([ys, xs]).astype(np.float32)) if a > 2 else np.eye(2)
        ev = np.sort(np.linalg.eigvalsh(cov)); el = np.sqrt(max(ev[1], 1e-3) / max(ev[0], 1e-3))
        p, im = pm[c], img[c].astype(np.float32)
        inside = im[m].mean(); around = im[ring].mean() if ring.any() else inside
        rows.append(np.array([np.log(a), p[m].mean(), p[m].max(), (p[m] > 0.8).mean(), ps[c][m].mean(), el, inside,
                              around - inside, np.hypot(xs.mean() + x0 - W / 2, ys.mean() + y0 - H / 2) / (W / 2),
                              p[ring].mean() if ring.any() else 0, dist_big[c][m].min(), np.log1p(n_big), src_t], np.float32))
    return np.array(rows, np.float32).reshape(-1, len(FEATS))

def candidate_sets(P, img):
    """-> dict source -> (label map, area per candidate); 'B' = B instances >= 50 px, t = faint blobs at p > t outside B(400)."""
    labB = instances('B', B_CFG, P)
    areaB = np.bincount(labB.ravel(), minlength=labB.max() + 1)[1:]
    big = np.isin(labB, np.where(areaB >= 400)[0] + 1)
    dist_big = cv2.distanceTransform((~big).astype(np.uint8), cv2.DIST_L2, 3)
    block = cv2.dilate(big.astype(np.uint8), K5) > 0
    out = {'B': labB}
    for t in T_LOW:
        n, lab, st, _ = cv2.connectedComponentsWithStats(((P[0] > t * 255) & ~block).astype(np.uint8), connectivity=8)
        keep = np.zeros(n, np.int32); ok = np.where(st[1:, cv2.CC_STAT_AREA] >= MIN_EXTRA)[0] + 1
        keep[ok] = np.arange(1, len(ok) + 1); out[t] = keep[lab]
    return out, dist_big, int(big.any() and len(np.unique(labB[big])))





import timm
B_CFG = EXP['B_CFG']; T_LOW = EXP['T_LOW']; MIN_EXTRA = EXP['MIN_EXTRA']; assert FEATS == EXP['FEATS']
W_CNN, THR, RT, RR = 0.6, 0.4, 0.5, 256
JUDGES = [('r34_128', 'resnet34', 128), ('cnxt_128', 'convnext_tiny', 128), ('effb3_128', 'efficientnet_b3', 128),
          ('r50_128', 'resnet50', 128), ('r34_192', 'resnet34', 192)]
T2 = np.load(WDIR + 'stage2_filter_table.npz')
m_f = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(T2['X'], T2['y'])
print('GBM filter refit on', T2['X'].shape, flush=True)

def load_fp16(model, path):
    sd = torch.load(path, map_location=DEV)
    model.load_state_dict({n: (v.float() if torch.is_floating_point(v) else v) for n, v in sd.items()}); return model.eval()

seg_models = []
for enc, tag in MEMBERS:
    CFG['encoder'] = enc
    for k in range(CFG['n_folds']):
        seg_models.append(load_fp16(make_model(pretrained=False), WDIR + f'{tag}/fold{k}.pth'))
judges = {}
for name, arch, size in JUDGES:
    judges[(name, size)] = [load_fp16(timm.create_model(arch, pretrained=False, in_chans=7, num_classes=1).to(DEV), WDIR + f'cnn_judge/{name}/fold{k}.pth') for k in range(5)]
refiners = [load_fp16(smp.Unet('resnet34', encoder_weights=None, in_channels=4, classes=1).to(DEV), WDIR + f'refine/fold{k}.pth') for k in range(5)]
print(len(seg_models), 'U-Nets,', sum(len(v) for v in judges.values()), 'CNN judges,', len(refiners), 'refiners loaded', flush=True)

def judge_crops(masks, img, prob, S):
    """7-channel crops: local window (1.5 x bbox + 32, >= 64 px) image / prob / mask; 512-px context image / prob / mask / others."""
    allm = np.zeros((H, W), bool)
    for m in masks: allm |= m
    out = []
    for m in masks:
        ys, xs = np.nonzero(m); cy, cx = ys.mean(), xs.mean()
        side = max(max(ys.max() - ys.min(), xs.max() - xs.min()) * 1.5 + 32, 64)
        def cut(a, sd, fill, interp):
            h = int(sd // 2); x0, y0 = int(cx) - h, int(cy) - h; pad = h + 2
            ap = cv2.copyMakeBorder(a, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=fill)
            return cv2.resize(ap[y0 + pad:y0 + pad + 2 * h, x0 + pad:x0 + pad + 2 * h], (S, S), interpolation=interp)
        mm = m.astype(np.uint8) * 255; oth = (allm & ~m).astype(np.uint8) * 255
        out.append(np.stack([cut(img, side, 128, cv2.INTER_AREA), cut(prob, side, 0, cv2.INTER_AREA), cut(mm, side, 0, cv2.INTER_AREA),
                             cut(img, 512, 128, cv2.INTER_AREA), cut(prob, 512, 0, cv2.INTER_AREA), cut(mm, 512, 0, cv2.INTER_AREA),
                             cut(oth, 512, 0, cv2.INTER_AREA)]))
    return np.stack(out)

@torch.no_grad()
def judge_scores(masks, img, prob):
    mem = []
    for (name, S), ms in judges.items():
        x = torch.from_numpy(judge_crops(masks, img, prob, S)).to(DEV).float() / 255; acc = 0
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=DEV == 'cuda'):
            for m in ms:
                for q in range(4):
                    acc = acc + torch.sigmoid(m(torch.rot90(x, q, (2, 3)))[:, 0].float())
        mem.append((acc / (4 * len(ms))).cpu().numpy())
    return np.mean(mem, 0)

def _geom(m):
    ys, xs = np.where(m); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(max(y1 - y0, x1 - x0) * 1.3 + 48); return (y0 + y1) // 2 - side // 2, (x0 + x1) // 2 - side // 2, side
def _crop(a, oy, ox, side):
    o = np.zeros((side, side), a.dtype); ya, xa = max(oy, 0), max(ox, 0); yb, xb = min(oy + side, H), min(ox + side, W)
    o[ya - oy:yb - oy, xa - ox:xb - ox] = a[ya:yb, xa:xb]; return o
_rs = lambda a: cv2.resize(a, (RR, RR), interpolation=cv2.INTER_AREA)

@torch.no_grad()
def refine(masks, img, prob):
    """Boundary refiner: re-draw every kept filament; conflicts resolved by area (larger instance keeps the pixels)."""
    allm = np.zeros((H, W), np.uint8)
    for m in masks: allm |= m.astype(np.uint8)
    xs, gs = [], []
    for m in masks:
        oy, ox, side = _geom(m); m8 = m.astype(np.uint8)
        xs.append(np.stack([_rs(_crop(img, oy, ox, side)), _rs(_crop(prob, oy, ox, side)), _rs(_crop(m8 * 255, oy, ox, side)),
                            _rs(_crop((allm & (1 - m8)) * 255, oy, ox, side))])); gs.append((oy, ox, side))
    x = torch.from_numpy(np.stack(xs)).to(DEV).float() / 255; acc = 0
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=DEV == 'cuda'):
        for m_ in refiners:
            for rr in range(4):
                for fl in (0, 1):
                    xx = torch.rot90(x, rr, (2, 3)); xx = xx.flip(3) if fl else xx
                    p = m_(xx).float().sigmoid(); p = p.flip(3) if fl else p; acc = acc + torch.rot90(p, -rr, (2, 3))
    PR = (acc[:, 0] / (8 * len(refiners))).cpu().numpy(); out = []
    for m, (oy, ox, side), pr in zip(masks, gs, PR):
        p = cv2.resize(pr, (side, side), interpolation=cv2.INTER_LINEAR) >= RT
        full = np.zeros((H, W), bool); ya, xa = max(oy, 0), max(ox, 0); yb, xb = min(oy + side, H), min(ox + side, W)
        full[ya:yb, xa:xb] = p[ya - oy:yb - oy, xa - ox:xb - ox]; out.append(full if full.any() else m)
    order = np.argsort([-m.sum() for m in masks]); taken = np.zeros((H, W), bool)
    for a in order: out[a] = out[a] & ~taken; taken |= out[a]
    return out

def run_image(path):
    """One test image -> list of boolean instance masks (final, non-overlapping) and a dict of intermediate results."""
    img = preprocess(path)
    P = (np.mean([predict(mdl, img) for mdl in seg_models], 0) * 255).astype(np.uint8)
    sets, dist_big, n_big = candidate_sets(P, img)
    labB = sets['B']; XB = cand_features(labB, P, img, dist_big, n_big, 0.6); keepB = np.exp(XB[:, 0]) >= 400
    labE = sets[0.3]; XE = cand_features(labE, P, img, dist_big, n_big, 0.3)
    masks = [labB == j for j in np.where(keepB)[0] + 1] + [labE == j for j in range(1, labE.max() + 1)]
    if not masks: return [], dict(img=img, P=P, cand=[], score=np.zeros(0))
    Xall = np.concatenate([XB[keepB], XE])
    score = W_CNN * judge_scores(masks, img, P[0]) + (1 - W_CNN) * m_f.predict(Xall)
    kept = [m for m, s in zip(masks, score) if s >= THR]
    final = refine(kept, img, P[0]) if kept else []
    return [m for m in final if m.any()], dict(img=img, P=P, cand=masks, score=score, kept=kept)

if os.environ.get('RUN_ALL', '1') == '1':   # the demo notebook sets RUN_ALL=0 to import the functions only
    test_files = sorted(os.listdir(TEST_DIR)); rows, t0 = [], time.time()
    for n, fn in enumerate(test_files):
        final, _ = run_image(TEST_DIR + fn)
        for q, m in enumerate(final):
            rows.append((f'{os.path.splitext(fn)[0]}_{q + 1}', mu.encode(np.asfortranarray(m.astype(np.uint8)))['counts'].decode()))
        if n % 20 == 0: print(f'{n + 1}/{len(test_files)} images, {len(rows)} filaments, {time.time() - t0:.0f}s', flush=True)
    sub = pd.DataFrame(rows, columns=['filament_id', 'segmentation_rle']); sub.to_csv(ARGS.out, index=False)
    print(f'{len(sub)} filaments on {sub.filament_id.str.rsplit("_", n=1).str[0].nunique()} of {len(test_files)} test images, '
          f'{(time.time() - t0) / len(test_files):.2f}s per image -> ' + ARGS.out)
