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
