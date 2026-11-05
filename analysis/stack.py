# Stacking: a small fully-convolutional "judge" learns, per pixel, how to combine the members' OOF probability maps
# (+ the image itself). Residual design: output logit = logit(mean of members) + learned correction (last layer zero-init),
# so it starts exactly at equal averaging. Trained per fold on OTHER folds' OOF maps -> honest OOF stacked maps.
import os, sys, time, pickle, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import GroupKFold
MEM = os.environ.get('MEMBERS', 'cnxt,effb4,regy').split(','); TAG = os.environ.get('STAG', 'stack3')
EPOCHS, SPE, CROP, BS = int(os.environ.get('EPOCHS', 10)), 400, 512, 8
R = '/root/autodl-tmp/'; C = R + 'cache/'; OUT = R + f'runs/m_{TAG}/'; os.makedirs(OUT, exist_ok=True)
MAPS = [np.load(R + f'runs/m_{m}/oof_maps.npy', mmap_mode='r') for m in MEM]
IMG, MSK, SPN = (np.load(C + n + '.npy', mmap_mode='r') for n in ['img', 'mask', 'spine'])
OBS = pickle.load(open(C + 'obs.pkl', 'rb')); N, H, W = IMG.shape
months = np.array([fn[:6] for fn in OBS]); folds = np.zeros(N, int)
for k, (_, va) in enumerate(GroupKFold(5).split(OBS, groups=months)):
    folds[va] = k
print('members', MEM, '| obs per fold', np.bincount(folds), flush=True)
DEV = 'cuda'; torch.manual_seed(0); np.random.seed(0)
CIN = 2 * len(MEM) + 1

class Judge(nn.Module):
    def __init__(s, cin, ch=32):
        super().__init__()
        s.body = nn.Sequential(nn.Conv2d(cin, ch, 3, padding=1), nn.GELU(),
                               nn.Conv2d(ch, ch, 3, padding=2, dilation=2), nn.GELU(),
                               nn.Conv2d(ch, ch, 3, padding=4, dilation=4), nn.GELU(),
                               nn.Conv2d(ch, ch, 3, padding=8, dilation=8), nn.GELU(),
                               nn.Conv2d(ch, ch, 3, padding=16, dilation=16), nn.GELU())
        s.head = nn.Conv2d(ch, 2, 1); nn.init.zeros_(s.head.weight); nn.init.zeros_(s.head.bias)
    def forward(s, x):                       # x: [B, CIN, h, w] in 0..1; channels: m1 mask, m1 spine, m2 mask, ..., image
        mm = x[:, :-1:2].mean(1, keepdim=True).clamp(1e-4, 1 - 1e-4); ms = x[:, 1:-1:2].mean(1, keepdim=True).clamp(1e-4, 1 - 1e-4)
        base = torch.cat([torch.logit(mm), torch.logit(ms)], 1)
        return base + s.head(s.body(x))

def stack_input(i, sl=np.s_[:, :]):
    parts = []
    for M in MAPS:
        parts += [M[i, 0][sl], M[i, 1][sl]]
    parts.append(IMG[i][sl])
    return np.stack(parts).astype(np.float32) / 255

def soft_dice(p, t, eps=1.0):
    return 1 - (2 * (p * t).sum((2, 3)) + eps) / (p.sum((2, 3)) + t.sum((2, 3)) + eps)

def sample(idx):
    i = np.random.choice(idx); m = MSK[i]
    if np.random.rand() < 0.7:
        ys, xs = np.nonzero(m[::8, ::8])
        if len(ys):
            j = np.random.randint(len(ys)); cy, cx = ys[j] * 8, xs[j] * 8
        else:
            cy, cx = np.random.randint(H), np.random.randint(W)
    else:
        cy, cx = np.random.randint(H), np.random.randint(W)
    y0, x0 = int(np.clip(cy - CROP // 2, 0, H - CROP)), int(np.clip(cx - CROP // 2, 0, W - CROP))
    sl = np.s_[y0:y0 + CROP, x0:x0 + CROP]
    x = stack_input(i, sl); t = np.stack([MSK[i][sl], SPN[i][sl]]).astype(np.float32) / 255
    r = np.random.randint(4); x, t = np.rot90(x, r, (1, 2)), np.rot90(t, r, (1, 2))
    if np.random.rand() < 0.5: x, t = x[:, :, ::-1], t[:, :, ::-1]
    return np.ascontiguousarray(x), np.ascontiguousarray(t)

OM = np.lib.format.open_memmap(OUT + 'oof_maps.npy', mode='w+', dtype=np.uint8, shape=(N, 2, H, W))
t00 = time.time()
for k in range(5):
    tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
    net = Judge(CIN).to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = EPOCHS * SPE // BS; sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.1)
    t0 = time.time(); net.train(); run = 0
    for st in range(steps):
        b = [sample(tr) for _ in range(BS)]
        x = torch.from_numpy(np.stack([a for a, _ in b])).to(DEV); t = torch.from_numpy(np.stack([c for _, c in b])).to(DEV)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = net(x)
        out = out.float(); p = out.sigmoid()
        loss = F.binary_cross_entropy_with_logits(out[:, :1], t[:, :1]) + soft_dice(p[:, :1], t[:, :1]).mean() \
               + 0.5 * F.binary_cross_entropy_with_logits(out[:, 1:], t[:, 1:])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); run = 0.98 * run + 0.02 * loss.item()
        if (st + 1) % (steps // 4) == 0:
            print(f'  fold {k} step {st + 1}/{steps} loss {run:.4f} [{time.time() - t0:.0f}s]', flush=True)
    net.eval()
    with torch.no_grad():
        for i in va:
            x = torch.from_numpy(stack_input(i))[None].to(DEV)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                p = net(x).float().sigmoid()[0]
            OM[i] = (p.cpu().numpy() * 255).round().astype(np.uint8)
    torch.save(net.state_dict(), OUT + f'judge_fold{k}.pth')
    print(f'fold {k} done [{time.time() - t0:.0f}s]', flush=True)
OM.flush(); del OM
print(f'STACK DONE {time.time() - t00:.0f}s', flush=True)
