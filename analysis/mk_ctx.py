s = open('/root/autodl-tmp/run.py').read()
def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a[:70]); s = s.replace(a, b)
# model: 3 image views + 2 context channels
rep("            x = x.float()\n            mu = self.blur(x, self.k_large)",
    "            x = x.float(); ctx = x[:, 1:]; x = x[:, :1]\n            mu = self.blur(x, self.k_large)")
rep("x3 = torch.cat([x, self.blur(x, self.k_small), lcn], 1)", "x3 = torch.cat([x, self.blur(x, self.k_small), lcn, ctx], 1)")
rep("in_channels=3, classes=2)).to(DEV)", "in_channels=5, classes=2)).to(DEV)")
# training crops: add the matching crop of the half-resolution context maps (OOF predictions of the global model)
rep("import sys, os, time, pickle, random, numpy as np, torch,", "import cv2, sys, os, time, pickle, random, numpy as np, torch,")
rep("IMG, MSK, SPN = (np.load(SHM + n + '.npy', mmap_mode='r') for n in ['img', 'mask', 'spine'])\n",
    "IMG, MSK, SPN = (np.load(SHM + n + '.npy', mmap_mode='r') for n in ['img', 'mask', 'spine'])\nCTX = np.load('/root/autodl-tmp/runs/lowres/ctx_oof.npy', mmap_mode='r')\n")
rep("""        r = np.random.randint(4)
        x, t = np.rot90(x, r), np.rot90(t, r, axes=(1, 2))
        if np.random.rand() < 0.5:
            x, t = x[:, ::-1], t[:, :, ::-1]
        x = (x - 0.5) * np.random.uniform(0.85, 1.15) + 0.5 + np.random.uniform(-0.05, 0.05)
        return torch.from_numpy(np.ascontiguousarray(x[None])), torch.from_numpy(np.ascontiguousarray(t))""",
"""        cy0, cx0, cc = y0 // 2, x0 // 2, c // 2
        ctx = CTX[i][:, cy0:cy0 + cc, cx0:cx0 + cc].astype(np.float32) / 255
        ctx = np.stack([cv2.resize(z, (c, c), interpolation=cv2.INTER_LINEAR) for z in ctx])
        r = np.random.randint(4)
        x, t, ctx = np.rot90(x, r), np.rot90(t, r, axes=(1, 2)), np.rot90(ctx, r, axes=(1, 2))
        if np.random.rand() < 0.5:
            x, t, ctx = x[:, ::-1], t[:, :, ::-1], ctx[:, :, ::-1]
        x = (x - 0.5) * np.random.uniform(0.85, 1.15) + 0.5 + np.random.uniform(-0.05, 0.05)
        return torch.from_numpy(np.ascontiguousarray(np.concatenate([x[None], ctx]))), torch.from_numpy(np.ascontiguousarray(t))""")
# inference: image + upsampled context
rep("""    x = torch.from_numpy(img_u8.astype(np.float32) / 255)[None, None].to(DEV)
    with torch.autocast('cuda', dtype=torch.float16):
        p = model(x).sigmoid() + model(x.flip(-1)).sigmoid().flip(-1)
    return (p[0] / 2).float().cpu().numpy()
""", """    a = img_u8.astype(np.float32) / 255 if img_u8.ndim == 2 else img_u8
    x = (torch.from_numpy(a)[None, None] if a.ndim == 2 else torch.from_numpy(a)[None]).to(DEV)
    with torch.autocast('cuda', dtype=torch.float16):
        p = model(x).sigmoid() + model(x.flip(-1)).sigmoid().flip(-1)
    return (p[0] / 2).float().cpu().numpy()

CTX_OOF = np.load('/root/autodl-tmp/runs/lowres/ctx_oof.npy', mmap_mode='r')
CTX_TEST = dict(zip(pickle.load(open('/root/autodl-tmp/runs/lowres/ctx_test_files.pkl', 'rb')), np.load('/root/autodl-tmp/runs/lowres/ctx_test.npy')))
def with_ctx(img, ctx):
    up = np.stack([cv2.resize(np.asarray(z), (W, H), interpolation=cv2.INTER_LINEAR) for z in ctx]).astype(np.float32) / 255
    return np.concatenate([img[None].astype(np.float32) / 255, up])
""")
rep("OOF[i] = (predict(model, DATA[i][0]) * 255).astype(np.uint8)", "OOF[i] = (predict(model, with_ctx(DATA[i][0], CTX_OOF[i])) * 255).astype(np.uint8)")
rep("predict(mdl, img) for mdl in models", "predict(mdl, with_ctx(img, CTX_TEST[fn])) for mdl in models", 2)
rep("if ARGS.smoke:\n", "_d = f'/root/autodl-tmp/runs/m_{ARGS.tag}/'; os.makedirs(_d, exist_ok=True)\n_mm = np.lib.format.open_memmap(_d + 'oof_maps.npy', mode='w+', dtype=np.uint8, shape=(len(OBS), 2, H, W))\nfor _i in OOF: _mm[_i] = OOF[_i]\n_mm.flush(); del _mm; print('saved OOF maps', flush=True)\nif ARGS.smoke:\n")
open('/root/autodl-tmp/run_ctx.py', 'w').write(s)
import ast; ast.parse(s); print('run_ctx ok', len(s))
