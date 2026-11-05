# Stage B of the two-resolution model: a GLOBAL-context U-Net trained on the WHOLE solar disk at half resolution (1024 px).
# Outputs (mask, spine) probability maps used as 2 extra input channels of the full-resolution model (stage A, run_ctx.py).
# usage: python lowres.py --folds 0,2,4   (run two processes in parallel: --folds 0,2,4 and --folds 1,3), then --finish
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
import os, sys, time, glob, pickle, argparse, numpy as np, cv2, torch, torch.nn.functional as F
import segmentation_models_pytorch as smp
from sklearn.model_selection import GroupKFold
ap = argparse.ArgumentParser(); ap.add_argument('--folds', default='0,1,2,3,4'); ap.add_argument('--encoder', default='tu-regnety_032')
ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--batch', type=int, default=4); ap.add_argument('--finish', action='store_true')
ARGS = ap.parse_args()
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
R = '/root/autodl-tmp/'; C = R + 'cache/'; OUT = R + 'runs/lowres/'; os.makedirs(OUT, exist_ok=True)
src = open(R + 'run.py').read()
H = W = 2048; S = 1024; DEV = 'cuda'
CFG = dict(encoder=ARGS.encoder, cldice_w=0.2, cldice_iters=8, spine_loss_w=0.5)
exec(src[src.index("MODEL_CODE = r'''") + len("MODEL_CODE = r'''"): src.index("'''\nexec(MODEL_CODE)")])     # MultiView, loss_fn
exec(src[src.index('# ---------- 1) preprocessing'): src.index('# ---------- 2) targets')])               # disk_mask, preprocess
OBS = pickle.load(open(C + 'obs.pkl', 'rb')); N = len(OBS)
months = np.array([fn[:6] for fn in OBS]); folds = np.zeros(N, int)
for k, (_, va) in enumerate(GroupKFold(5).split(OBS, groups=months)):
    folds[va] = k
down = lambda a: cv2.resize(a, (S, S), interpolation=cv2.INTER_AREA)
t0 = time.time()
I2, M2, P2 = (np.load(C + n + '.npy', mmap_mode='r') for n in ['img', 'mask', 'spine'])
IMG = np.stack([down(np.array(I2[i])) for i in range(N)]); MSK = np.stack([down(np.array(M2[i])) for i in range(N)])
SPN = np.stack([down(np.array(P2[i])) for i in range(N)])
J = glob.glob(R + 'data/**/test_images', recursive=True)[0] + '/'; TEST = sorted(os.listdir(J))
print(f'half-res data ready {time.time() - t0:.0f}s | obs per fold {np.bincount(folds)} | test {len(TEST)}', flush=True)

@torch.no_grad()
def predict(model, img):
    model.eval(); x = torch.from_numpy(img.astype(np.float32) / 255)[None, None].to(DEV)
    with torch.autocast('cuda', dtype=torch.float16):
        p = model(x).sigmoid() + model(x.flip(-1)).sigmoid().flip(-1)
    return (p[0] / 2).float().cpu().numpy()

def batch(idx):
    xs, ts = [], []
    for i in np.random.choice(idx, ARGS.batch):
        x = IMG[i].astype(np.float32) / 255; t = np.stack([MSK[i], SPN[i]]).astype(np.float32) / 255
        r = np.random.randint(4); x, t = np.rot90(x, r), np.rot90(t, r, (1, 2))
        if np.random.rand() < 0.5: x, t = x[:, ::-1], t[:, :, ::-1]
        x = (x - 0.5) * np.random.uniform(0.85, 1.15) + 0.5 + np.random.uniform(-0.05, 0.05)
        xs.append(np.ascontiguousarray(x[None])); ts.append(np.ascontiguousarray(t))
    return torch.from_numpy(np.stack(xs)).to(DEV), torch.from_numpy(np.stack(ts)).to(DEV)

if not ARGS.finish:
    TESTIMG = None
    for k in [int(v) for v in ARGS.folds.split(',')]:
        torch.manual_seed(42 + k); np.random.seed(42 + k)
        tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
        model = make_model(True); opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        spe = len(tr) // ARGS.batch; steps = ARGS.epochs * spe
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=0.1); scaler = torch.amp.GradScaler('cuda')
        t0 = time.time()
        for ep in range(ARGS.epochs):
            model.train(); lm_s = ls_s = 0
            for _ in range(spe):
                x, t = batch(tr)
                with torch.autocast('cuda', dtype=torch.float16):
                    out = model(x)
                loss, lm, ls = loss_fn(out.float(), t)
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
                lm_s += lm; ls_s += ls
            if (ep + 1) % 5 == 0:
                print(f'  fold {k} epoch {ep + 1}/{ARGS.epochs} mask {lm_s / spe:.4f} spine {ls_s / spe:.4f} [{time.time() - t0:.0f}s]', flush=True)
        torch.save(model.state_dict(), OUT + f'fold{k}.pth')
        oof = np.stack([(predict(model, IMG[i]) * 255).round().astype(np.uint8) for i in va])
        np.save(OUT + f'oof_fold{k}.npy', oof)
        if TESTIMG is None:
            TESTIMG = np.stack([down(preprocess(J + fn)) for fn in TEST])
        np.save(OUT + f'test_fold{k}.npy', np.stack([(predict(model, im) * 255).round().astype(np.uint8) for im in TESTIMG]))
        print(f'fold {k} done [{time.time() - t0:.0f}s]', flush=True)
else:
    CO = np.zeros((N, 2, S, S), np.uint8)
    for k in range(5):
        CO[np.where(folds == k)[0]] = np.load(OUT + f'oof_fold{k}.npy')
    np.save(OUT + 'ctx_oof.npy', CO)
    CT = np.mean([np.load(OUT + f'test_fold{k}.npy').astype(np.float32) for k in range(5)], 0).round().astype(np.uint8)
    np.save(OUT + 'ctx_test.npy', CT); pickle.dump(TEST, open(OUT + 'ctx_test_files.pkl', 'wb'))
    # quick quality check of the global model alone: pixel Dice of mask at half resolution vs consensus>0.5
    p, t = CO[:, 0] > 127, MSK > 127
    print(f'LOWRES FINISH: OOF pixel Dice (mask>0.5 vs consensus>0.5, 1024 px) {2 * (p & t).sum() / (p.sum() + t.sum()):.4f}', flush=True)
