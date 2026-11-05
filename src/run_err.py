# Solar filament pipeline (port of the Kaggle notebook, model v2 = V16) for a single-GPU AutoDL machine.
# usage: python run.py --encoder resnet34 --tag r34 [--epochs 40 --spe 720 --per_gpu 2 --smoke]
import os, sys, argparse, glob, json, pickle
ap = argparse.ArgumentParser()
ap.add_argument('--encoder', default='resnet34'); ap.add_argument('--tag', required=True)
ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--spe', type=int, default=720)
ap.add_argument('--batch', type=int, default=8); ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--per_gpu', type=int, default=1); ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--smoke', action='store_true'); ap.add_argument('--eval_only', action='store_true')
ap.add_argument('--ensemble', default='')   # e.g. tu-convnext_tiny:cnxt,efficientnet-b4:effb4 -> average their probability maps
ARGS = ap.parse_args()
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
import matplotlib; matplotlib.use('Agg')
from pycocotools import mask as mu
J = glob.glob('/root/autodl-tmp/data/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json', recursive=True)[0]
ROOT = J.rsplit('/train/', 1)[0]
d = json.load(open(J))
anns = {}
_ = [anns.setdefault(a['image_id'], []).append(a) for a in d['annotations']]
print('annotations:', J, '| images:', len(d['images']), '| filaments:', len(d['annotations']))


# ===================== Solar filament instance segmentation: mask + spine U-Net, 5-fold CV =====================
# One network, two outputs: ch0 = filament mask (what we submit), ch1 = spine line (auxiliary training task,
# and also used by post-processing B to re-join fragments). Targets = soft consensus over all annotators.
# 5 folds grouped by month -> out-of-fold (OOF) score on all 707 training observations; test = mean of the 5 models.
import os, json, time, random, shutil, cv2, numpy as np, pandas as pd, torch, torch.nn.functional as F
import segmentation_models_pytorch as smp
from collections import defaultdict
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from skimage.morphology import skeletonize

CFG = dict(seed=ARGS.seed, n_folds=5, crop=768, batch=ARGS.batch, epochs=25, samples_per_epoch=480, lr=ARGS.lr,
           encoder=ARGS.encoder, spine_width=5, spine_loss_w=0.5, num_workers=4,
           train=False)   # True: train the 5 fold models (~50 min on 2xT4); False: load saved weights, no re-training
H = W = 2048
IMG_DIR, TEST_DIR, OUT = ROOT + '/train/train_images/', ROOT + '/test/test_images/', f'/root/autodl-tmp/runs/{ARGS.tag}/'
DEV = 'cuda'

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

seed_all(CFG['seed'])
print(torch.__version__, smp.__version__, torch.cuda.get_device_name(0), '| GPUs:', torch.cuda.device_count())

os.makedirs(OUT, exist_ok=True)
print('run dir', OUT, '| encoder', CFG['encoder'])


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

# ---------- 2) targets: soft consensus of all annotators (pycocotools rasterisation = same as the official scorer) ----------
def poly_mask(poly):
    return mu.decode(mu.frPyObjects([poly.ravel().tolist()], H, W))[..., 0]

def as_xy(a):
    return np.asarray(a, dtype=np.float64).reshape(-1, 2)

def build_obs(fn, view_list):
    """-> (image u8, mask target u8, spine target u8, filament centres). Targets = mean over annotators x 255."""
    m = np.zeros((H, W), np.float32); s = np.zeros((H, W), np.float32); centres = []
    for v in view_list:
        mv = np.zeros((H, W), np.uint8); sv = np.zeros((H, W), np.uint8)
        for poly, sp in v:
            mv |= poly_mask(poly)
            if len(sp) >= 2:
                cv2.polylines(sv, [np.round(sp).astype(np.int32)], False, 1, CFG['spine_width'])
            centres.append(poly.mean(0))
        m += mv; s += sv
    k = 255.0 / len(view_list)
    return preprocess(IMG_DIR + fn), (m * k).astype(np.uint8), (s * k).astype(np.uint8), np.array(centres)

views = defaultdict(list)            # observation file -> list of annotator views -> list of (polygon, spine)
for im_ in d['images']:
    views[im_['file_name']].append([(as_xy(a['segmentation'][0]), as_xy(a['spine'])) for a in anns.get(im_['id'], [])])
OBS = sorted(views)

t0 = time.time()
CACHE = '/root/autodl-tmp/cache/'
if os.path.exists(CACHE + 'done'):
    _I, _M, _S = (np.load(CACHE + n + '.npy', mmap_mode='r') for n in ['img', 'mask', 'spine'])
    _C = pickle.load(open(CACHE + 'cen.pkl', 'rb'))
    assert pickle.load(open(CACHE + 'obs.pkl', 'rb')) == OBS
    DATA = [(np.array(_I[i]), np.array(_M[i]), np.array(_S[i]), _C[i]) for i in range(len(OBS))]
    print('loaded preprocessed observations from cache')
else:
    DATA = Parallel(n_jobs=12)(delayed(build_obs)(fn, views[fn]) for fn in OBS)
    os.makedirs(CACHE, exist_ok=True)
    for j, n in enumerate(['img', 'mask', 'spine']):
        np.save(CACHE + n + '.npy', np.stack([x[j] for x in DATA]))
    pickle.dump([x[3] for x in DATA], open(CACHE + 'cen.pkl', 'wb')); pickle.dump(OBS, open(CACHE + 'obs.pkl', 'wb'))
    open(CACHE + 'done', 'w').write('ok')
print(f'{len(DATA)} observations prepared in {time.time() - t0:.0f}s')

# ---------- 3) 5 folds grouped by month (filaments persist for days-weeks, so same-month images are near-duplicates) ----------
months = np.array([fn[:6] for fn in OBS])
folds = np.zeros(len(OBS), int)
for k, (_, va) in enumerate(GroupKFold(CFG['n_folds']).split(OBS, groups=months)):
    folds[va] = k
print('observations per fold:', np.bincount(folds).tolist(),
      '| annotator views per fold:', [sum(len(views[OBS[i]]) for i in np.where(folds == k)[0]) for k in range(CFG['n_folds'])])


# ---------- 4) training samples: random 768x768 crops at FULL resolution (filaments are only a few px wide) ----------
class CropDS(torch.utils.data.Dataset):
    def __init__(self, idx, n, crop):
        self.idx, self.n, self.c = idx, n, crop

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        img, m, s, cen = DATA[np.random.choice(self.idx)]
        c = self.c
        if len(cen) and np.random.rand() < 0.7:              # 70%: window around a random filament centre, 30%: anywhere
            cx, cy = cen[np.random.randint(len(cen))] + np.random.randint(-c // 3, c // 3, 2)
        else:
            cx, cy = np.random.randint(c // 2, W - c // 2, 2)
        x0, y0 = int(np.clip(cx - c // 2, 0, W - c)), int(np.clip(cy - c // 2, 0, H - c))
        x = img[y0:y0 + c, x0:x0 + c].astype(np.float32) / 255
        t = np.stack([m[y0:y0 + c, x0:x0 + c], s[y0:y0 + c, x0:x0 + c]]).astype(np.float32) / 255
        k = np.random.randint(4)                               # rotations + flips
        x, t = np.rot90(x, k), np.rot90(t, k, axes=(1, 2))
        if np.random.rand() < 0.5:
            x, t = x[:, ::-1], t[:, :, ::-1]
        x = (x - 0.5) * np.random.uniform(0.85, 1.15) + 0.5 + np.random.uniform(-0.05, 0.05)   # contrast / brightness jitter
        return torch.from_numpy(np.ascontiguousarray(x[None])), torch.from_numpy(np.ascontiguousarray(t))

def seed_worker(_):
    np.random.seed(torch.initial_seed() % 2 ** 32)

# ---------- 5) model + loss ----------
def soft_dice(p, t, eps=1.0):
    return 1 - ((2 * (p * t).sum((2, 3)) + eps) / (p.sum((2, 3)) + t.sum((2, 3)) + eps)).mean()

def loss_fn(logits, t):
    """BCE + soft Dice per channel; the spine channel is an auxiliary task weighted by spine_loss_w."""
    lm = F.binary_cross_entropy_with_logits(logits[:, :1], t[:, :1]) + soft_dice(logits[:, :1].sigmoid(), t[:, :1])
    ls = F.binary_cross_entropy_with_logits(logits[:, 1:], t[:, 1:]) + soft_dice(logits[:, 1:].sigmoid(), t[:, 1:])
    return lm + CFG['spine_loss_w'] * ls, lm.item(), ls.item()

def make_model(pretrained=True):
    return smp.Unet(CFG['encoder'], encoder_weights='imagenet' if pretrained else None, in_channels=1, classes=2).to(DEV)

print('params (M):', round(sum(p.numel() for p in make_model(False).parameters()) / 1e6, 1))


# ---------- 5b) model v2: 3-view input + clDice auxiliary loss + ~2.4x longer training (overrides cell 5) ----------
# Error analysis: missed filament parts are mainly LOW-CONTRAST (only ~5% darker than the background) and thin.
#  * the network now sees 3 views of the same image, computed inside the model (identical for training and test):
#    raw (limb-flattened) image, a lightly denoised copy, and a local-contrast-normalised copy (faint features amplified)
#  * clDice (centre-line Dice) as an auxiliary mask loss: rewards covering the whole centre line of thin structures
#  * 40 epochs x 720 crops per fold (was 25 x 480)
CFG.update(train=True, epochs=ARGS.epochs, samples_per_epoch=ARGS.spe, cldice_w=0.2, cldice_iters=8)
import gc, subprocess, pickle, psutil


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

print('model v2 params (M):', round(sum(p.numel() for p in make_model(False).parameters()) / 1e6, 1),
      '| epochs', CFG['epochs'], 'x', CFG['samples_per_epoch'], 'crops | clDice weight', CFG['cldice_w'])

# ---------- training in separate processes, one fold per GPU ----------
# Runs 1-2 died in fold 3 with "out of memory": the RAM log of run 3 showed the notebook process growing by ~0.11 GB per
# epoch (~1.2 MB per training step) during training with nn.DataParallel, independent of data-loader workers -> it would
# run out of RAM again. Fix: each fold is trained in its OWN short-lived Python process on ONE GPU (no DataParallel),
# two folds at a time (GPU 0 and GPU 1). Whatever a process accumulates is released when it exits. Same model, loss,
# crops, augmentation, optimiser and schedule; batch 8 on one GPU instead of 4 + 4 on two.
# The notebook then loads the saved fold weights (CFG['train'] = False) for the out-of-fold and test predictions.
SHM = OUT + 'fold_data/'
os.makedirs(SHM, exist_ok=True)
for j, name in enumerate(['img', 'mask', 'spine']):
    mm = np.lib.format.open_memmap(SHM + name + '.npy', mode='w+', dtype=np.uint8, shape=(len(DATA), H, W))
    for i in range(len(DATA)):
        mm[i] = DATA[i][j]
    mm.flush(); del mm
pickle.dump(dict(CFG=CFG, folds=np.asarray(folds), cen=[np.asarray(d[3]) for d in DATA], H=H, W=W, OUT=OUT),
            open(SHM + 'meta.pkl', 'wb'))

FOLD_SCRIPT = r"""
import sys, os, time, pickle, random, numpy as np, torch, torch.nn.functional as F, segmentation_models_pytorch as smp, psutil
k = int(sys.argv[1]); SHM = sys.argv[2]; DEV = 'cuda'
torch.set_num_threads(2)
meta = pickle.load(open(SHM + 'meta.pkl', 'rb')); CFG, folds, CEN, H, W, OUT = (meta[x] for x in ['CFG', 'folds', 'cen', 'H', 'W', 'OUT'])
IMG, MSK, SPN = (np.load(SHM + n + '.npy', mmap_mode='r') for n in ['img', 'mask', 'spine'])
random.seed(CFG['seed'] + k); np.random.seed(CFG['seed'] + k); torch.manual_seed(CFG['seed'] + k)

class CropDS(torch.utils.data.Dataset):
    def __init__(self, idx, n, crop):
        self.idx, self.n, self.c = idx, n, crop
    def __len__(self):
        return self.n
    def __getitem__(self, _):
        i = np.random.choice(self.idx); img, m, s, cen = IMG[i], MSK[i], SPN[i], CEN[i]
        c = self.c
        if len(cen) and np.random.rand() < 0.7:
            cx, cy = cen[np.random.randint(len(cen))] + np.random.randint(-c // 3, c // 3, 2)
        else:
            cx, cy = np.random.randint(c // 2, W - c // 2, 2)
        x0, y0 = int(np.clip(cx - c // 2, 0, W - c)), int(np.clip(cy - c // 2, 0, H - c))
        x = img[y0:y0 + c, x0:x0 + c].astype(np.float32) / 255
        t = np.stack([m[y0:y0 + c, x0:x0 + c], s[y0:y0 + c, x0:x0 + c]]).astype(np.float32) / 255
        r = np.random.randint(4)
        x, t = np.rot90(x, r), np.rot90(t, r, axes=(1, 2))
        if np.random.rand() < 0.5:
            x, t = x[:, ::-1], t[:, :, ::-1]
        x = (x - 0.5) * np.random.uniform(0.85, 1.15) + 0.5 + np.random.uniform(-0.05, 0.05)
        return torch.from_numpy(np.ascontiguousarray(x[None])), torch.from_numpy(np.ascontiguousarray(t))
""" + MODEL_CODE + r"""
model = make_model()
dl = torch.utils.data.DataLoader(CropDS(np.where(folds != k)[0], CFG['samples_per_epoch'], CFG['crop']),
                                 batch_size=CFG['batch'], num_workers=4, drop_last=True, persistent_workers=True,
                                 worker_init_fn=lambda w: np.random.seed((torch.initial_seed() + w) % 2 ** 32))
opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG['lr'], total_steps=CFG['epochs'] * len(dl), pct_start=0.1)
scaler = torch.amp.GradScaler('cuda'); t0 = time.time()
for ep in range(CFG['epochs']):
    model.train(); lm_sum = ls_sum = 0
    for x, t in dl:
        x, t = x.to(DEV, non_blocking=True), t.to(DEV, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16):
            out = model(x)
        loss, lm, ls = loss_fn(out.float(), t)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        lm_sum += lm; ls_sum += ls
    if (ep + 1) % 5 == 0:
        print(f'  fold {k} epoch {ep + 1:2d}/{CFG["epochs"]}  mask loss {lm_sum / len(dl):.4f}  spine loss {ls_sum / len(dl):.4f}'
              f'  [{time.time() - t0:.0f}s, process RAM {psutil.Process().memory_info().rss / 2**30:.1f} GB]', flush=True)
torch.save(model.state_dict(), f'{OUT}unet_mask_spine_fold{k}.pth.tmp')
os.replace(f'{OUT}unet_mask_spine_fold{k}.pth.tmp', f'{OUT}unet_mask_spine_fold{k}.pth')
print(f'  fold {k} done in {time.time() - t0:.0f}s', flush=True)
"""
open(SHM + 'train_fold.py', 'w').write(FOLD_SCRIPT)

def run_folds_parallel(fold_ids, n_gpu):
    queue, running, logs, done = list(fold_ids), {}, {}, []
    while queue or running:
        for g in range(n_gpu):
            if g not in running and queue:
                k = queue.pop(0); logs[k] = open(f'{SHM}fold{k}.log', 'w')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g % max(torch.cuda.device_count(), 1)))
                running[g] = (k, subprocess.Popen([sys.executable, SHM + 'train_fold.py', str(k), SHM], env=env,
                                                  stdout=logs[k], stderr=subprocess.STDOUT), 0)
        time.sleep(20)
        for g in list(running):
            k, p, shown = running[g]
            lines = open(f'{SHM}fold{k}.log').read().splitlines()
            for line in lines[shown:]:
                print(line, flush=True)
            running[g] = (k, p, len(lines))
            if p.poll() is not None:
                del running[g]; logs[k].close()
                if p.returncode != 0:
                    raise RuntimeError(f'fold {k} training process failed (exit {p.returncode}):\n' + '\n'.join(lines[-30:]))
                done.append(k)
    vm = psutil.virtual_memory()
    print(f'trained folds {sorted(done)} | notebook process RAM {psutil.Process().memory_info().rss / 2**30:.1f} GB,'
          f' available {vm.available / 2**30:.1f} of {vm.total / 2**30:.1f} GB')

if ARGS.eval_only:
    CFG['train'] = False                  # weights already in OUT: load them, skip training
if CFG['train']:
    make_model(pretrained=True)            # download the ImageNet weights once, before the fold processes start
    run_folds_parallel(range(CFG['n_folds']), max(1, torch.cuda.device_count()) * ARGS.per_gpu)
    CFG['train'] = False                  # cell 6 now loads the weights just trained (saved in /kaggle/working)

import shutil as _sh; _sh.rmtree(SHM, ignore_errors=True)   # free the scratch copies


# ---------- 6) 5-fold models + out-of-fold (OOF) predictions: each observation is predicted by a model that never saw it ----------
# CFG['train'] = True trains the 5 fold models; False loads the saved weights, so results are reproduced without re-training.
@torch.no_grad()
def predict(model, img_u8):
    """2048x2048 inference with horizontal-flip TTA -> float32 [2, H, W] = (mask prob, spine prob)."""
    model.eval()
    x = torch.from_numpy(img_u8.astype(np.float32) / 255)[None, None].to(DEV)
    with torch.autocast('cuda', dtype=torch.float16):
        p = model(x).sigmoid() + model(x.flip(-1)).sigmoid().flip(-1)
    return (p[0] / 2).float().cpu().numpy()

def train_fold(k):
    seed_all(CFG['seed'] + k)
    model = make_model()
    net = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    dl = torch.utils.data.DataLoader(CropDS(np.where(folds != k)[0], CFG['samples_per_epoch'], CFG['crop']), batch_size=CFG['batch'],
                                     num_workers=CFG['num_workers'], drop_last=True, worker_init_fn=seed_worker, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG['lr'], total_steps=CFG['epochs'] * len(dl), pct_start=0.1)
    scaler = torch.amp.GradScaler('cuda')
    t0 = time.time()
    for ep in range(CFG['epochs']):
        net.train(); lm_sum = ls_sum = 0
        for x, t in dl:
            x, t = x.to(DEV, non_blocking=True), t.to(DEV, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16):
                out = net(x)
            loss, lm, ls = loss_fn(out.float(), t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            lm_sum += lm; ls_sum += ls
        if (ep + 1) % 5 == 0:
            print(f'  fold {k} epoch {ep + 1:2d}/{CFG["epochs"]}  mask loss {lm_sum / len(dl):.4f}  spine loss {ls_sum / len(dl):.4f}  [{time.time() - t0:.0f}s]')
    torch.save(model.state_dict(), f'{OUT}unet_mask_spine_fold{k}.pth')
    return model

def find_input(name):
    """Locate a file anywhere under /kaggle/input (os.walk follows symlinked input mounts, unlike recursive glob)."""
    for root, _, files in os.walk('/kaggle/input', followlinks=True):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(f'{name} not found under /kaggle/input; inputs: {os.listdir("/kaggle/input")}')

def load_fold(k):
    path = f'{OUT}unet_mask_spine_fold{k}.pth'
    if not os.path.exists(path):          # weights of an earlier run, attached to this notebook as input data
        shutil.copy(find_input(f'unet_mask_spine_fold{k}.pth'), path)   # keep a copy in this run's output
    model = make_model(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=DEV))
    return model

ENS = [m.split(':') for m in ARGS.ensemble.split(',')] if ARGS.ensemble else []
def load_member(enc, tag, k):
    CFG['encoder'] = enc
    m = make_model(pretrained=False)
    m.load_state_dict(torch.load(f'/root/autodl-tmp/runs/{tag}/unet_mask_spine_fold{k}.pth', map_location=DEV))
    return m
if ENS: print('ensemble members:', ENS)

OOF = {}                              # observation index -> uint8 [2, H, W] (mask prob, spine prob) x 255
for k in range(CFG['n_folds']):
    MEMS = [load_member(e, t, k) for e, t in ENS] if ENS else [train_fold(k) if CFG['train'] else load_fold(k)]
    t0 = time.time()
    for i in np.where(folds == k)[0]:
        OOF[i] = (np.mean([predict(m, DATA[i][0]) for m in MEMS], 0) * 255).astype(np.uint8)
    print(f'fold {k}: OOF predictions for {np.sum(folds == k)} observations, {(time.time() - t0) / np.sum(folds == k):.2f}s per image')
    del MEMS; torch.cuda.empty_cache()


if ENS:
    _mm = np.lib.format.open_memmap(OUT + 'oof_maps.npy', mode='w+', dtype=np.uint8, shape=(len(OBS), 2, H, W))
    for _i in OOF: _mm[_i] = OOF[_i]
    _mm.flush(); del _mm; print('saved OOF maps', flush=True)
if ARGS.smoke:
    print('SMOKE TEST OK', {i: OOF[i].shape for i in list(OOF)[:2]}); sys.exit(0)

# ---------- 7) evaluation (official-style PQ: one prediction set vs EVERY annotator, IoU > 0.5) + post-processing A ----------
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

def gt_view(view):
    """One annotator view -> list of (bbox slice, boolean crop), one per filament (cropping keeps matching fast)."""
    out = []
    for poly, _ in view:
        g = poly_mask(poly)
        ys, xs = np.nonzero(g)
        if len(ys):
            sl = np.s_[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            out.append((sl, g[sl] > 0))
    return out

def match_view(lab, parea, gts):
    """PQ bookkeeping for one annotator view (IoU > 0.5 => match, as in the official self-evaluation notebook)."""
    n = len(parea) - 1
    iou_sum, tp, matched, dices, o2m = 0.0, 0, set(), [], []
    gts_per_pred = np.zeros(n + 1, int)
    for sl, g in gts:
        inter = np.bincount(lab[sl][g], minlength=n + 1); inter[0] = 0
        hits = np.nonzero(inter)[0]
        o2m.append(len(hits)); gts_per_pred[hits] += 1
        if len(hits):
            j = hits[np.argmax(inter[hits])]
            ga = g.sum(); iou = inter[j] / (ga + parea[j] - inter[j])
            if iou > 0.5:
                tp += 1; iou_sum += iou; matched.add(j); dices.append(2 * inter[j] / (ga + parea[j]))
    return iou_sum, tp, n - len(matched), len(gts) - tp, dices, o2m, gts_per_pred[1:].tolist()

def evaluate_inst(make_lab, idx):
    """Pooled PQ over the observations idx for an instance builder make_lab(i) -> label map."""
    S = np.zeros(4); dices, o2m, m2o = [], [], []
    for i in idx:
        lab = make_lab(i)
        parea = np.bincount(lab.ravel(), minlength=lab.max() + 1)
        for gts in GT[i]:
            r = match_view(lab, parea, gts)
            S += r[:4]; dices += r[4]; o2m += r[5]; m2o += r[6]
    iou_sum, tp, fp, fn = S
    den = max(tp + 0.5 * fp + 0.5 * fn, 1)
    return dict(PQ=iou_sum / den, SQ=iou_sum / max(tp, 1), RQ=tp / den, TP=int(tp), FP=int(fp), FN=int(fn),
                dice=float(np.mean(dices)) if dices else 0.0, o2m=np.array(o2m, int), m2o=np.array(m2o, int))

def show(name, cfg, r):
    print(f'{name}', cfg, {k: round(float(r[k]), 4) for k in ('PQ', 'SQ', 'RQ')}, 'TP/FP/FN', r['TP'], r['FP'], r['FN'])

ALL = np.arange(len(OBS))
t0 = time.time()
GT = {i: [gt_view(v) for v in views[OBS[i]]] for i in ALL}
print(f'GT masks for {len(GT)} observations ready in {time.time() - t0:.0f}s')

def cfg_of(key):
    """RES key (method, thr, lo, min_area, link) -> keyword arguments of the post-processing."""
    m, thr, lo, ma, link = key
    return dict(thr=thr, min_area=ma, lo=lo) if m == 'A' else dict(thr=thr, min_area=ma, lo=lo, link=link)

RES = {}                              # (method, thr, lo, min_area, link) -> metrics on all OOF predictions
best = lambda cond: max((k for k in RES if cond(k)), key=lambda k: RES[k]['PQ'])

def run_A(thr, ma, lo=None):
    key = ('A', thr, lo, ma, 0)
    RES[key] = r = evaluate_inst(lambda i: to_instances(OOF[i][0], **cfg_of(key)), ALL); show('A', cfg_of(key), r)

for thr in [0.5, 0.6, 0.7]:                                  # plain threshold
    run_A(thr, 400)
_, bthr, _, _, _ = best(lambda k: k[0] == 'A')
for ma in [300, 600]:
    run_A(bthr, ma)
_, bthr, _, bma, _ = best(lambda k: k[0] == 'A')
for lo in [0.2, 0.3, 0.4]:                                   # hysteresis: same detection threshold, wider outline
    run_A(bthr, bma, lo)
_, _, blo, _, _ = best(lambda k: k[0] == 'A' and k[2] is not None)
run_A(round(min(bthr + 0.1, 0.9), 2), bma, blo)                       # with wider outlines a stricter detection threshold may pay off
run_A(bthr, bma + 200, blo)                                  # ... and so may a larger minimum area
A_PLAIN, A_HYST = best(lambda k: k[0] == 'A' and k[2] is None), best(lambda k: k[0] == 'A' and k[2] is not None)
print('best A (plain):', cfg_of(A_PLAIN), round(float(RES[A_PLAIN]['PQ']), 4), '| best A (hysteresis):', cfg_of(A_HYST), round(float(RES[A_HYST]['PQ']), 4))


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


# ---------- 9) B on all OOF predictions (on top of the best plain and the best hysteresis A), choose the final post-processing ----------
for base in [A_PLAIN, A_HYST]:
    for link in [30, 40]:
        key = ('B', base[1], base[2], base[3], link)
        RES[key] = r = evaluate_inst(lambda i: instances('B', cfg_of(key), OOF[i]), ALL); show('B', cfg_of(key), r)
VARIANTS = {'A': A_PLAIN, 'A + hysteresis': A_HYST,
            'B': best(lambda k: k[0] == 'B' and k[2] is None), 'B + hysteresis': best(lambda k: k[0] == 'B' and k[2] is not None)}
FINAL = max(VARIANTS.values(), key=lambda k: RES[k]['PQ'])
METHOD, CFG_PP = FINAL[0], cfg_of(FINAL)

rows = []
for name, key in VARIANTS.items():
    pf = [evaluate_inst(lambda i: instances(key[0], cfg_of(key), OOF[i]), np.where(folds == k)[0])['PQ'] for k in range(CFG['n_folds'])]
    r = RES[key]
    rows.append({'variant': name, 'config': str(cfg_of(key)), 'OOF PQ': r['PQ'], 'SQ': r['SQ'], 'RQ': r['RQ'], 'TP': r['TP'], 'FP': r['FP'], 'FN': r['FN'],
                 **{f'fold {k}': v for k, v in enumerate(pf)}, 'fold mean': np.mean(pf), 'fold std': np.std(pf, ddof=1)})
summary = pd.DataFrame(rows)
print(summary.drop(columns='config').round(4).to_string(index=False))
print('\nconfigs:', {n: cfg_of(k) for n, k in VARIANTS.items()})
print(f'final post-processing: {METHOD} {CFG_PP}  (OOF PQ {RES[FINAL]["PQ"]:.4f})')
json.dump({'oof': {str(k): {m: round(float(v), 4) for m, v in RES[k].items() if m in ('PQ', 'SQ', 'RQ', 'TP', 'FP', 'FN', 'dice')} for k in RES},
           'final': [METHOD, CFG_PP], 'summary': summary.round(4).to_dict('records')}, open(OUT + 'oof_results.json', 'w'), indent=1)


# ---------- 11) test: mean of the 5 fold models -> final post-processing -> submission.csv (COCO RLE, column-major) ----------
models = []
for k in range(CFG['n_folds']):
    if ENS: break
    mdl = make_model(pretrained=False)
    mdl.load_state_dict(torch.load(f'{OUT}unet_mask_spine_fold{k}.pth', map_location=DEV))
    models.append(mdl)
if ENS: models = [load_member(e, t, k) for e, t in ENS for k in range(CFG['n_folds'])]
rows, t0 = [], time.time()
test_files = sorted(os.listdir(TEST_DIR))
for fn in test_files:
    img = preprocess(TEST_DIR + fn)
    P = (np.mean([predict(mdl, img) for mdl in models], 0) * 255).astype(np.uint8)
    lab = instances(METHOD, CFG_PP, P)
    for k in range(1, lab.max() + 1):
        rle = mu.encode(np.asfortranarray((lab == k).astype(np.uint8)))['counts'].decode()
        rows.append((f'{os.path.splitext(fn)[0]}_{k}', rle))
sub = pd.DataFrame(rows, columns=['filament_id', 'segmentation_rle'])
sub.to_csv(OUT + 'submission.csv', index=False)
print(f'{(time.time() - t0) / len(test_files):.2f}s per test image (5-model ensemble + post-processing {METHOD})')
print(sub.shape, '|', sub.filament_id.str.rsplit('_', n=1).str[0].nunique(), 'of', len(test_files), 'test images with >=1 prediction')


# ---------- 13) small & faint filaments: more candidates + learned false-positive filter (no U-Net re-training) ----------
# Error analysis: ~460 annotated filaments < 400 px are deleted by min_area, ~470 of 400-1000 px are never detected (too faint).
# Idea: let MORE candidates through and let the instance filter (gradient boosting, trained out-of-fold) decide:
#   (i)  smaller min_area for the B instances (threshold 0.6), or
#   (ii) keep B (min_area 400) and ADD faint blobs = connected components of p > t_low (t_low < 0.6) that do not touch any
#        B instance (existing filaments keep exactly their shape; only new, separate candidates are added).
# Every configuration gets its own out-of-fold filter; all are scored with the official-style PQ on the 707 OOF images.
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

def match_table(lab, gts_views):
    """-> IOU [n, views], GID [n, views]: best IoU of every candidate with any GT of each annotator view."""
    n = lab.max(); parea = np.bincount(lab.ravel(), minlength=n + 1)
    IOU, GID = np.zeros((n, len(gts_views)), np.float32), np.full((n, len(gts_views)), -1, np.int32)
    for v, gts in enumerate(gts_views):
        for gi, (sl, g) in enumerate(gts):
            inter = np.bincount(lab[sl][g], minlength=n + 1); inter[0] = 0
            hits = np.nonzero(inter)[0]
            if len(hits):
                iou = inter[hits] / (g.sum() + parea[hits] - inter[hits])
                better = iou > IOU[hits - 1, v]
                IOU[hits[better] - 1, v] = iou[better]; GID[hits[better] - 1, v] = gi
    return IOU, GID

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

# ---- 0) why are the small / faint filaments missed?  peak U-Net probability inside the annotated filaments that B misses ----
t0 = time.time(); C = {}; miss_p = []
for i in ALL:
    sets, dist_big, n_big = candidate_sets(OOF[i], DATA[i][0])
    C[i] = {}
    for src, lab in sets.items():
        X = cand_features(lab, OOF[i], DATA[i][0], dist_big, n_big, 0.6 if src == 'B' else src)
        IOU, GID = match_table(lab, GT[i])
        C[i][src] = dict(X=X, IOU=IOU, GID=GID)
    labB, aB = sets['B'], C[i]['B']['X'][:, 0]
    for gts in GT[i]:                                     # annotated filaments not touched by any B(400) instance
        for sl, g in gts:
            hit = labB[sl][g]; hit = hit[hit > 0]
            if not len(hit) or np.exp(aB[np.unique(hit) - 1]).max() < 400:
                miss_p.append((g.sum(), OOF[i][0][sl][g].max() / 255))
NGT = {i: [len(g) for g in GT[i]] for i in ALL}
print(f'candidate tables for {len(C)} observations in {time.time() - t0:.0f}s')
mp = pd.DataFrame(miss_p, columns=['area', 'peak p'])
mp['size'] = pd.cut(mp.area, [0, 400, 1000, 1e9], labels=['< 400 px', '400-1000 px', '>= 1000 px'])
mp['peak'] = pd.cut(mp['peak p'], [-0.01, 0.1, 0.3, 0.4, 0.5, 0.6, 1.01], labels=['< 0.1', '0.1-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '>= 0.6'])
print('\nannotated filaments not covered by any B instance (>= 400 px): peak U-Net probability inside them')
print(pd.crosstab(mp['size'], mp['peak'], margins=True).to_string())

# ---- 1) configurations: candidate pool -> out-of-fold filter -> PQ ----
def pool(i, cfg):
    kind, val = cfg
    b = C[i]['B']; keepB = np.exp(b['X'][:, 0]) >= (val if kind == 'min_area' else 400)
    parts = [(b['X'][keepB], b['IOU'][keepB], b['GID'][keepB])]
    if kind == 'faint':
        e = C[i][val]; parts.append((e['X'], e['IOU'], e['GID']))
    return [np.concatenate(z) for z in zip(*parts)]

def pq_keep(cfg, keep_fn, idx=ALL):
    S = np.zeros(4)
    for i in idx:
        X, IOU, GID = pool(i, cfg); kept = np.where(keep_fn(i, X))[0]
        for v, ngt in enumerate(NGT[i]):
            iou, gid = IOU[kept, v], GID[kept, v]; ok = iou > 0.5; best = {}
            for a, b in zip(gid[ok], iou[ok]):
                best[a] = max(best.get(a, 0), b)
            tp = len(best); S += [sum(best.values()), tp, len(kept) - tp, ngt - tp]
    return S[0] / max(S[1] + 0.5 * S[2] + 0.5 * S[3], 1), S

gbm = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
CFGS = [('min_area', 400), ('min_area', 250), ('min_area', 150), ('min_area', 80)] + [('faint', t) for t in T_LOW]
RESF, rows = {}, []
for cfg in CFGS:
    own = np.concatenate([np.full(len(pool(i, cfg)[0]), i) for i in ALL])
    X = np.concatenate([pool(i, cfg)[0] for i in ALL]); y = np.concatenate([(pool(i, cfg)[1] > 0.5).mean(1) for i in ALL])
    sc = np.zeros(len(y))
    for k in range(CFG['n_folds']):
        tr, va = folds[own] != k, folds[own] == k
        sc[va] = gbm().fit(X[tr], y[tr]).predict(X[va])
    SC = {i: sc[own == i] for i in ALL}
    sweep = {s: pq_keep(cfg, lambda i, X_, s=s: SC[i] >= s)[0] for s in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]}
    s_best = max(sweep, key=sweep.get)
    no_f = pq_keep(cfg, lambda i, X_: np.ones(len(X_), bool))[0]
    _, S = pq_keep(cfg, lambda i, X_: SC[i] >= s_best)
    pf = [pq_keep(cfg, lambda i, X_: SC[i] >= s_best, np.where(folds == k)[0])[0] for k in range(CFG['n_folds'])]
    RESF[cfg] = dict(X=X, y=y, s=s_best, pf=pf, SC=SC)
    rows.append({'candidates': f'B min_area {cfg[1]}' if cfg[0] == 'min_area' else f'B(400) + faint blobs p > {cfg[1]}',
                 '#cand': len(y), 'PQ no filter': no_f, 'PQ + filter': sweep[s_best], 'keep s >=': s_best,
                 'TP': int(S[1]), 'FP': int(S[2]), 'FN': int(S[3]), **{f'fold {k}': v for k, v in enumerate(pf)}})
tabS = pd.DataFrame(rows)
ref = np.array(RESF[CFGS[0]]['pf'])
tabS['folds better than row 1'] = [int((np.array(RESF[c]['pf']) > ref).sum()) for c in CFGS]
print('\n' + tabS.round(4).to_string(index=False))
BEST_C = CFGS[int(tabS['PQ + filter'].values.argmax())]
print(f'chosen: {BEST_C}  (row 1 = the current submission: B + filter, here with crop-based features)')

# ===================== ERROR ANALYSIS of the chosen configuration (ens) =====================
import collections
ED = OUT + 'err/'; os.makedirs(ED, exist_ok=True)
pickle.dump(dict(C=C, RESF={k: dict(s=v['s'], SC=v['SC']) for k, v in RESF.items()}, BEST_C=BEST_C), open(ED + 'state.pkl', 'wb'))
cfg = BEST_C; s_thr = RESF[cfg]['s']; SCb = RESF[cfg]['SC']
print(f'\n===== ERROR ANALYSIS: {cfg}, keep score >= {s_thr} =====', flush=True)
site = lambda fn: os.path.splitext(fn)[0][-2:]
KR = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
fp_rows, gt_rows, tp_rows, img_rows = [], [], [], []
t0 = time.time()
for i in ALL:
    X, IOU, GID = pool(i, cfg); kept = np.where(SCb[i] >= s_thr)[0]
    img = DATA[i][0]; pm = OOF[i][0]
    # full label map per view (for annotator agreement)
    vlabs = []
    for gts in GT[i]:
        L = np.zeros((H, W), np.int32)
        for gi, (sl, g) in enumerate(gts):
            L[sl][g] = gi + 1
        vlabs.append(L)
    S_img = np.zeros(4)
    for v, gts in enumerate(GT[i]):
        iou, gid = IOU[kept, v], GID[kept, v]
        # greedy as in pq_keep: best pred per GT among iou>0.5
        best = {}
        for j, (a, b) in enumerate(zip(gid, iou)):
            if b > 0.5 and b > best.get(a, (None, 0))[1]:
                best[a] = (j, b)
        matched_pred = {j for j, _ in best.values()}
        S_img += [sum(b for _, b in best.values()), len(best), len(kept) - len(best), len(gts) - len(best)]
        # other views' agreement for each GT of this view
        others = [w for w in range(len(GT[i])) if w != v]
        for gi, (sl, g) in enumerate(gts):
            ga = g.sum()
            agree = np.nan
            if others:
                agree = 0
                for w in others:
                    inter = np.bincount(vlabs[w][sl][g], minlength=len(GT[i][w]) + 1); inter[0] = 0
                    if inter.any():
                        k_ = inter.argmax(); gb = (vlabs[w] == k_).sum()
                        agree += inter[k_] / (ga + gb - inter[k_]) > 0.5
                agree = agree / len(others)
            # contrast: preprocessed image inside vs 10-px ring
            y0, y1 = max(sl[0].start - 12, 0), min(sl[0].stop + 12, H); x0, x1 = max(sl[1].start - 12, 0), min(sl[1].stop + 12, W)
            m = np.zeros((y1 - y0, x1 - x0), np.uint8); m[sl[0].start - y0:sl[0].stop - y0, sl[1].start - x0:sl[1].stop - x0] = g
            ring = (cv2.dilate(m, KR) > 0) & (m == 0)
            im = img[y0:y1, x0:x1].astype(np.float32)
            contrast = (np.median(im[ring]) - np.median(im[m > 0])) / 255 if ring.any() else np.nan
            ys, xs = np.nonzero(g)
            cov = np.cov(np.stack([ys, xs]).astype(np.float32)) if len(ys) > 2 else np.eye(2)
            ev = np.sort(np.linalg.eigvalsh(cov)); length = 4 * np.sqrt(max(ev[1], 0)); width = ga / max(length, 1)
            hit = gid == gi
            n_touch = int((hit & (iou > 0)).sum())
            biou = float(iou[hit].max()) if hit.any() else 0.0
            if gi in best:
                status = 'TP'; j = best[gi][0]
                tp_rows.append(dict(i=i, v=v, iou=best[gi][1], gt_area=ga, pred_area=float(np.exp(X[kept[j], 0])), contrast=contrast, agree=agree))
            elif n_touch == 0:
                status = 'untouched'
            elif n_touch >= 2 and biou <= 0.5:
                status = 'fragmented'
            elif biou >= 0.3:
                status = 'near miss'
            else:
                status = 'weak overlap'
            gt_rows.append(dict(i=i, v=v, fold=folds[i], site=site(OBS[i]), n_views=len(GT[i]), area=ga, length=length, width=width,
                                contrast=contrast, peak_p=pm[sl][g].max() / 255, mean_p=pm[sl][g].mean() / 255, agree=agree,
                                best_iou=biou, n_touch=n_touch, status=status,
                                radius=np.hypot(xs.mean() + sl[1].start - W / 2, ys.mean() + sl[0].start - H / 2) / (W / 2)))
        # false positives of this view
        for j in range(len(kept)):
            if j in matched_pred:
                continue
            b = iou[j]
            # does it match another annotator's filament?
            other_match = any(IOU[kept[j], w] > 0.5 for w in others)
            if b > 0.5:
                cat = 'duplicate/lost tie'
            elif b >= 0.3:
                cat = 'near miss'
            elif b > 0:
                cat = 'weak overlap'
            else:
                cat = 'spurious'
            fp_rows.append(dict(i=i, v=v, site=site(OBS[i]), n_views=len(GT[i]), cat=cat, other_annotator_match=other_match,
                                area=float(np.exp(X[kept[j], 0])), mean_p=float(X[kept[j], 1]), score=float(SCb[i][kept[j]]),
                                contrast=float(X[kept[j], 7]), radius=float(X[kept[j], 8])))
    img_rows.append(dict(i=i, fn=OBS[i], site=site(OBS[i]), fold=folds[i], n_views=len(GT[i]), iou_sum=S_img[0], TP=S_img[1], FP=S_img[2], FN=S_img[3]))
print(f'tables built in {time.time() - t0:.0f}s', flush=True)
G, Fp, T, I = pd.DataFrame(gt_rows), pd.DataFrame(fp_rows), pd.DataFrame(tp_rows), pd.DataFrame(img_rows)
for n, d in [('gt', G), ('fp', Fp), ('tp', T), ('img', I)]:
    d.to_csv(ED + n + '.csv', index=False)
tp, fp, fn, isum = len(T), len(Fp), int((G.status != 'TP').sum()), T.iou.sum()
den = tp + 0.5 * fp + 0.5 * fn
print(f'TP {tp} FP {fp} FN {fn} | PQ {isum / den:.4f} SQ {isum / tp:.4f} RQ {tp / den:.4f}')
pq = lambda isum, tp, fp, fn: isum / (tp + 0.5 * fp + 0.5 * fn)
print('\n--- what-if (upper bounds) ---')
print(f'  no FP at all:                 PQ {pq(isum, tp, 0, fn):.4f}')
print(f'  no FN at all (IoU = SQ):      PQ {pq(isum + fn * isum / tp, tp + fn, fp, 0):.4f}')
print(f'  perfect IoU on TPs (SQ=1):    PQ {pq(tp, tp, fp, fn):.4f}')
nm = int((G.status == 'near miss').sum())
print(f'  all {nm} near-miss FN fixed (IoU 0.55, their FP gone): PQ {pq(isum + 0.55 * nm, tp + nm, max(fp - nm, 0), fn - nm):.4f}')
nsp = int((Fp.cat == 'spurious').sum())
print(f'  all {nsp} spurious FP removed: PQ {pq(isum, tp, fp - nsp, fn):.4f}')
print('\n--- FN by type ---'); print(G[G.status != 'TP'].status.value_counts().to_string())
print('\n--- FP by type ---'); print(Fp.cat.value_counts().to_string())
print('FP that match ANOTHER annotator\'s filament (disagreement):', int(Fp.other_annotator_match.sum()), f'of {fp}')
G['size'] = pd.cut(G.area, [0, 400, 1000, 3000, 1e9], labels=['<400', '400-1k', '1k-3k', '>=3k'])
G['contrast q'] = pd.qcut(G.contrast, 5, labels=['q1 faint', 'q2', 'q3', 'q4', 'q5 dark'])
G['det'] = G.status == 'TP'
print('\n--- recall (TP rate) by size x contrast quintile (n) ---')
print(G.pivot_table(index='size', columns='contrast q', values='det', aggfunc='mean', observed=False).round(2).to_string())
print(G.pivot_table(index='size', columns='contrast q', values='det', aggfunc='size', observed=False).to_string())
print('\n--- recall by annotator agreement (multi-annotator images only) ---')
ga_ = G[G.n_views > 1].copy(); ga_['agree b'] = pd.cut(ga_.agree, [-0.01, 0.01, 0.99, 1.01], labels=['nobody else', 'some', 'all others'])
print(ga_.groupby('agree b', observed=False).agg(n=('det', 'size'), recall=('det', 'mean')).round(3).to_string())
print('\n--- FN status by size ---'); print(pd.crosstab(G['size'], G.status).to_string())
print('\n--- TP quality ---')
T['ratio'] = T.pred_area / T.gt_area
T['size'] = pd.cut(T.gt_area, [0, 400, 1000, 3000, 1e9], labels=['<400', '400-1k', '1k-3k', '>=3k'])
print(T.groupby('size', observed=False).agg(n=('iou', 'size'), mean_iou=('iou', 'mean'), median_area_ratio=('ratio', 'median')).round(3).to_string())
print('TP IoU quantiles 10/25/50/75/90:', T.iou.quantile([.1, .25, .5, .75, .9]).round(3).tolist(),
      '| area ratio pred/gt quantiles:', T.ratio.quantile([.1, .25, .5, .75, .9]).round(2).tolist())
print('\n--- FP by size / mean p ---')
Fp['size'] = pd.cut(Fp.area, [0, 400, 1000, 3000, 1e9], labels=['<400', '400-1k', '1k-3k', '>=3k'])
print(pd.crosstab(Fp['size'], Fp.cat).to_string())
print('\n--- per site (GONG station code) ---')
I['PQ'] = I.iou_sum / (I.TP + 0.5 * I.FP + 0.5 * I.FN).clip(lower=1)
st = I.groupby('site').agg(images=('i', 'size'), iou=('iou_sum', 'sum'), TP=('TP', 'sum'), FP=('FP', 'sum'), FN=('FN', 'sum'))
st['PQ'] = st.iou / (st.TP + 0.5 * st.FP + 0.5 * st.FN); print(st.round(4).to_string())
print('\n--- by number of annotators on the image ---')
nv = I.groupby('n_views').agg(images=('i', 'size'), iou=('iou_sum', 'sum'), TP=('TP', 'sum'), FP=('FP', 'sum'), FN=('FN', 'sum'))
nv['PQ'] = nv.iou / (nv.TP + 0.5 * nv.FP + 0.5 * nv.FN); print(nv.round(4).to_string())
print('\n--- per-image PQ distribution ---', I.PQ.describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_dict())
print('worst 10 images:'); print(I.nsmallest(10, 'PQ')[['fn', 'site', 'n_views', 'TP', 'FP', 'FN', 'PQ']].round(3).to_string(index=False))
# human ceiling on multi-annotator images: annotator vs annotator PQ (each view predicts the others)
S_h = np.zeros(4)
for i in I[I.n_views > 1].i:
    for v in range(len(GT[i])):
        L = np.zeros((H, W), np.int32)
        for gi, (sl, g) in enumerate(GT[i][v]):
            L[sl][g] = gi + 1
        parea = np.bincount(L.ravel(), minlength=L.max() + 1)
        for w in range(len(GT[i])):
            if w != v:
                r = match_view(L, parea, GT[i][w]); S_h += r[:4]
mv = I[I.n_views > 1]
print(f'\nhuman ceiling (annotator vs other annotators, {len(mv)} multi-annotator images): PQ {S_h[0] / (S_h[1] + 0.5 * S_h[2] + 0.5 * S_h[3]):.4f}'
      f' | ours on the same images: PQ {mv.iou_sum.sum() / (mv.TP.sum() + 0.5 * mv.FP.sum() + 0.5 * mv.FN.sum()):.4f}')
print('ERROR ANALYSIS DONE', flush=True)
sys.exit(0)

# what did the chosen configuration recover?  matched annotated filaments by size (summed over annotators), vs row 1
GTA = {i: [np.array([g.sum() for _, g in gts]) for gts in GT[i]] for i in ALL}
def tp_sizes(cfg, scores):
    out = []
    for i in ALL:
        X, IOU, GID = pool(i, cfg); kept = scores[i] >= RESF[cfg]['s']
        for v in range(len(NGT[i])):
            ok = kept & (IOU[:, v] > 0.5)
            out += GTA[i][v][np.unique(GID[ok, v])].tolist()
    return pd.cut(pd.Series(out, dtype=float), [0, 400, 1000, 1e9], labels=['< 400 px', '400-1000 px', '>= 1000 px']).value_counts().sort_index()
print('\nmatched annotated filaments by size:')
print(pd.DataFrame({str(c): tp_sizes(c, RESF[c]['SC']) for c in dict.fromkeys([CFGS[0], BEST_C])}).to_string())

# ---- 2) test: models trained on all OOF candidates of the chosen configuration ----
m_f = gbm().fit(RESF[BEST_C]['X'], RESF[BEST_C]['y']); s_best = RESF[BEST_C]['s']
rows, t0, n_kept = [], time.time(), 0
for fn in test_files:
    img = preprocess(TEST_DIR + fn)
    P = (np.mean([predict(mdl, img) for mdl in models], 0) * 255).astype(np.uint8)
    sets, dist_big, n_big = candidate_sets(P, img)
    labB = sets['B']; XB = cand_features(labB, P, img, dist_big, n_big, 0.6)
    keepB = np.exp(XB[:, 0]) >= (BEST_C[1] if BEST_C[0] == 'min_area' else 400)
    layers = [(labB, np.where(keepB)[0] + 1, XB[keepB])]
    if BEST_C[0] == 'faint':
        labE = sets[BEST_C[1]]; XE = cand_features(labE, P, img, dist_big, n_big, BEST_C[1])
        layers.append((labE, np.arange(1, labE.max() + 1), XE))
    k_out = 0
    for lab, ids, X in layers:
        if not len(ids):
            continue
        keep = m_f.predict(X) >= s_best
        for j in ids[keep]:
            k_out += 1
            rle = mu.encode(np.asfortranarray((lab == j).astype(np.uint8)))['counts'].decode()
            rows.append((f'{os.path.splitext(fn)[0]}_{k_out}', rle))
    n_kept += k_out
if os.path.exists(OUT + 'submission.csv'):
    shutil.copy(OUT + 'submission.csv', OUT + 'submission_B.csv')
sub = pd.DataFrame(rows, columns=['filament_id', 'segmentation_rle'])
sub.to_csv(OUT + 'submission.csv', index=False)
print(f'test: {BEST_C} + filter -> {len(sub)} filaments on {sub.filament_id.str.rsplit("_", n=1).str[0].nunique()} of {len(test_files)} images, '
      f'{(time.time() - t0) / len(test_files):.2f}s per image | submission.csv written (plain B kept as submission_B.csv)')

# ---- export for a Kaggle inference notebook: filter training table + chosen configurations ----
np.savez_compressed(OUT + 'filter_export.npz', X=RESF[BEST_C]['X'], y=RESF[BEST_C]['y'])
json.dump(dict(encoder=ARGS.ensemble or CFG['encoder'], best_c=list(BEST_C), s_best=float(s_best), B_CFG=B_CFG, T_LOW=list(T_LOW),
               MIN_EXTRA=int(MIN_EXTRA), FEATS=FEATS, final_pp=[METHOD, CFG_PP]), open(OUT + 'filter_export.json', 'w'), indent=1)
print('exported filter table', RESF[BEST_C]['X'].shape)
