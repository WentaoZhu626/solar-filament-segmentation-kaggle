
"""Test submission: ens3 candidates -> 13-feature GBM filter (fit on all OOF candidates) blended 50/50 with the 5-fold CNN judge."""
import os, sys, time
_src = open('/root/autodl-tmp/analysis/vis_prep.py').read()
exec(_src[:_src.index('sc = np.zeros(len(y))')])          # builds C (OOF candidates), X, y, gbm, candidate_sets, cand_features ...
import torch, timm, pandas as pd
from pycocotools import mask as mu
sys.argv = ['x', 'none']; exec(open('/root/autodl-tmp/analysis/cnn_filter.py').read().split("if STAGE == 'crops':")[0])
W_CNN = float(os.environ.get('W_CNN', 0.5)); THR = float(os.environ.get('THR', 0.4))
m_f = gbm().fit(X, y); print('filter fit on', len(y), 'OOF candidates', flush=True)
TM = np.load(B + 'runs/err3/test_maps.npy', mmap_mode='r'); TI = np.load(B + 'runs/err3/test_img.npy', mmap_mode='r')
TEST_DIR_ = glob.glob(B + 'data/**/test_images/', recursive=True)[0]; files = sorted(os.listdir(TEST_DIR_)); assert len(files) == TM.shape[0]
# MEMBERS = "tag:arch:size,..." (tag '' = the first ResNet34 run); member scores are averaged
MEMBERS = [m.split(':') for m in os.environ.get('MEMBERS', ':resnet34:128').split(',')]
models = {}
for tag, arch, sz in MEMBERS:
    for k in range(5):
        m = timm.create_model(arch, pretrained=False, in_chans=7, num_classes=1).cuda().eval()
        m.load_state_dict(torch.load(OUT + f'cnn{tag}_fold{k}.pth', map_location='cuda')); models.setdefault((tag, int(sz)), []).append(m)
print('members', [(t, len(v)) for t, v in models.items()], flush=True)

import segmentation_models_pytorch as smp
RF = os.environ.get('RF_OUT', '/root/refine/'); RT = float(os.environ.get('RT', 0.5)); RR = 256
refs = []
for k in range(5):
    m_ = smp.Unet('resnet34', encoder_weights=None, in_channels=4, classes=1).cuda().eval(); m_.load_state_dict(torch.load(RF + f'ref_fold{k}.pth', map_location='cuda')); refs.append(m_)
def _geom(m):
    ys, xs = np.where(m); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(max(y1 - y0, x1 - x0) * 1.3 + 48); cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    return cy - side // 2, cx - side // 2, side
def _crop(a, oy, ox, side):
    out = np.zeros((side, side), a.dtype)
    ya, xa = max(oy, 0), max(ox, 0); yb, xb = min(oy + side, 2048), min(ox + side, 2048)
    out[ya - oy:yb - oy, xa - ox:xb - ox] = a[ya:yb, xa:xb]; return out
def _rs(a): return cv2.resize(a, (RR, RR), interpolation=cv2.INTER_AREA)
@torch.no_grad()
def refine_masks(masks, img, P0):
    allm = np.zeros((2048, 2048), np.uint8)
    for mm in masks: allm |= mm.astype(np.uint8)
    xs, gs = [], []
    for mm in masks:
        oy, ox, side = _geom(mm); m8 = mm.astype(np.uint8)
        xs.append(np.stack([_rs(_crop(img, oy, ox, side)), _rs(_crop(P0, oy, ox, side)), _rs(_crop(m8 * 255, oy, ox, side)), _rs(_crop((allm & (1 - m8)) * 255, oy, ox, side))])); gs.append((oy, ox, side))
    x = torch.from_numpy(np.stack(xs)).cuda().float() / 255; acc = 0
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for m_ in refs:
            for rr in range(4):
                for fl in (0, 1):
                    xx = torch.rot90(x, rr, (2, 3)); xx = xx.flip(3) if fl else xx
                    p = m_(xx).float().sigmoid(); p = p.flip(3) if fl else p; acc = acc + torch.rot90(p, -rr, (2, 3))
    PR = (acc[:, 0] / (8 * len(refs))).cpu().numpy()
    out = []
    for mm, (oy, ox, side), pr in zip(masks, gs, PR):
        p = cv2.resize(pr, (side, side), interpolation=cv2.INTER_LINEAR) >= RT
        full = np.zeros((2048, 2048), bool); ya, xa = max(oy, 0), max(ox, 0); yb, xb = min(oy + side, 2048), min(ox + side, 2048)
        full[ya:yb, xa:xb] = p[ya - oy:yb - oy, xa - ox:xb - ox]
        out.append(full if full.any() else mm)
    order = np.argsort([-mm.sum() for mm in masks]); taken = np.zeros((2048, 2048), bool)
    for a in order: out[a] = out[a] & ~taken; taken |= out[a]
    return out
rows_base, rows_new, rows_ref, stat = [], [], [], []
for jj, fn in enumerate(files):
    P = np.asarray(TM[jj]); img = np.asarray(TI[jj])
    sets, dist_big, n_big = candidate_sets(P, img)
    labB = sets['B']; XB = cand_features(labB, P, img, dist_big, n_big, 0.6); keepB = np.exp(XB[:, 0]) >= 400
    labE = sets[0.3]; XE = cand_features(labE, P, img, dist_big, n_big, 0.3)
    src = [('B', int(i) + 1) for i in np.where(keepB)[0]] + [(0.3, i + 1) for i in range(len(XE))]
    Xall = np.concatenate([XB[keepB], XE]) if len(XE) else XB[keepB]
    if not len(src): continue
    sf = m_f.predict(Xall)
    r = dict(B_png=cv2.imencode('.png', labB.astype(np.uint16))[1].tobytes(), F_png=cv2.imencode('.png', labE.astype(np.uint16))[1].tobytes(),
             prob_png=cv2.imencode('.png', P[0])[1].tobytes(), cand_src=src, kept=np.ones(len(src), bool))
    mem = []
    for (tag, sz), ms in models.items():
        S = sz; cr = torch.from_numpy(crops_for(r, img)).cuda().float() / 255
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            mem.append((sum(torch.sigmoid(m(torch.rot90(cr, q, (2, 3)))[:, 0].float()) for m in ms for q in range(4)) / (4 * len(ms))).cpu().numpy())
    sc = np.mean(mem, 0); final = W_CNN * sc + (1 - W_CNN) * sf
    stem = os.path.splitext(fn)[0]; kb = kn = 0; keepm = []
    for (s_, lid), f0, f1 in zip(src, sf, final):
        lab = labB if s_ == 'B' else labE
        if f0 >= 0.4 or f1 >= THR: mk = (lab == lid); rle = mu.encode(np.asfortranarray(mk.astype(np.uint8)))['counts'].decode()
        if f0 >= 0.4: kb += 1; rows_base.append((f'{stem}_{kb}', rle))
        if f1 >= THR: kn += 1; rows_new.append((f'{stem}_{kn}', rle)); keepm.append(mk)
    if keepm:
        for q, mk in enumerate(refine_masks(keepm, img, P[0])):
            if mk.any(): rows_ref.append((f'{stem}_{q + 1}', mu.encode(np.asfortranarray(mk.astype(np.uint8)))['counts'].decode()))
    stat.append((len(src), kb, kn))
    if jj % 30 == 0: print(jj, stat[-1], flush=True)
os.makedirs(B + 'runs/cnnf', exist_ok=True)
pd.DataFrame(rows_ref, columns=['filament_id', 'segmentation_rle']).to_csv(B + 'runs/cnnf/' + os.environ.get('REFNAME', 'submission_cnn_refine.csv'), index=False); print('refined rows', len(rows_ref))
sub = pd.DataFrame(rows_new, columns=['filament_id', 'segmentation_rle']); sub.to_csv(B + 'runs/cnnf/' + os.environ.get('SUBNAME', 'submission_cnn_blend.csv'), index=False)
st = np.array(stat); print(f'candidates {st[:, 0].sum()} | ens3 filter keeps {st[:, 1].sum()} (orig submission 1103) | CNN blend keeps {st[:, 2].sum()} on {sub.filament_id.str.rsplit("_", n=1).str[0].nunique()} images')
