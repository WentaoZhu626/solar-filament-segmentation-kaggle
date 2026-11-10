
# two-stage: existence decided by the validated keep rule on the original B/F candidates (vis kept flags = GBM rule,
# or CNN-judge rule), shape chosen among alternatives of each kept parent (original / refined / threshold components)
import pickle, numpy as np, os
CGD = os.environ.get('CGDIR', '/root/cg/'); D = pickle.load(open(CGD + 'hyp2.pkl', 'rb')); HY = D['H']; NGT = D['NGT']; folds = np.asarray(D['folds'])
V = pickle.load(open('/root/autodl-tmp/runs/vis/vis.pkl', 'rb')); REC = V['rec']
CO = '/root/autodl-tmp/runs/cnnf/'
cn = np.mean([np.load(CO + f'oof{t}.npy') for t in ['', '_cnxt', '_effb3', '_r50', '_r34_192']], 0)
gb = np.concatenate([r['score'] for r in REC]); offs = np.cumsum([0] + [len(r['kept']) for r in REC])
O = np.load(CGD + 'cnn_oof.npy'); co = np.cumsum([0] + [len(h['src']) for h in HY])
def run(rule, chooser, min_piou=0.3):
    S = np.zeros(4); per = []
    for i, h in enumerate(HY):
        nBF = h['nB'] + h['nF']
        if rule == 'gbm': keep = REC[i]['kept']
        else: keep = (0.6 * cn[offs[i]:offs[i + 1]] + 0.4 * gb[offs[i]:offs[i + 1]]) >= 0.4
        u_true = np.where(h['IOU'] > 0.5, h['IOU'], 0).mean(1); u_cnn = O[co[i]:co[i + 1], 1]
        edges = set(map(tuple, h['edges'])) | set((b, a) for a, b in h['edges'])
        chosen = []
        for j in np.where(keep)[0]:
            alts = [j] + [a for a in range(nBF, len(h['src'])) if h['par'][a] == j and h['piou'][a] >= min_piou]
            if chooser == 'orig': c = j
            elif chooser == 'refined': c = next((a for a in alts if h['src'][a] == 2), j)
            elif chooser == 'cnn': c = max(alts, key=lambda a: u_cnn[a])
            elif chooser == 'cnn_or': c = max([a for a in alts if h['src'][a] in (0, 1, 2)], key=lambda a: u_cnn[a])
            elif chooser == 'oracle': c = max(alts, key=lambda a: u_true[a])
            elif chooser == 'oracle_or': c = max([a for a in alts if h['src'][a] in (0, 1, 2)], key=lambda a: u_true[a])
            chosen.append((j, c))
        # resolve conflicts between chosen alternatives: larger parent keeps its choice, the other falls back to its parent
        area = h['area']; chosen.sort(key=lambda t: -area[t[0]]); fin = []
        for j, c in chosen:
            if any((c, d) in edges for d in fin): c = j
            if any((c, d) in edges for d in fin): continue
            fin.append(c)
        s = np.zeros(4)
        for v in range(h['IOU'].shape[1]):
            best = {}
            for c in fin:
                if h['IOU'][c, v] > 0.5:
                    g = h['GID'][c, v]; best[g] = max(best.get(g, 0), h['IOU'][c, v])
            tp = len(best); s += [sum(best.values()), tp, len(fin) - tp, NGT[i][v] - tp]
        S += s; per.append(s)
    per = np.array(per); pf = []
    for k in range(5):
        Sk = per[folds == k].sum(0); pf.append(round(Sk[0] / (Sk[1] + 0.5 * Sk[2] + 0.5 * Sk[3]), 4))
    return S[0] / (S[1] + 0.5 * S[2] + 0.5 * S[3]), S, pf, per
R = {}
for rule in ['gbm', 'cnn']:
    for ch in ['orig', 'refined', 'cnn_or', 'cnn', 'oracle_or', 'oracle']:
        p, S, pf, per = run(rule, ch); R[(rule, ch)] = per
        print(f'{rule:4s} {ch:10s} PQ {p:.4f} SQ {S[0] / S[1]:.4f} TP/FP/FN {S[1:].astype(int)} folds {pf}', flush=True)
pickle.dump(R, open(CGD + 'two_stage.pkl', 'wb')); print('TWO DONE')
