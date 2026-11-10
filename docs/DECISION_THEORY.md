# Decision theory for the keep rule

Script: `stage2/experiments/keep_threshold_theory.py`. Data: 9795 out-of-fold candidates, official bookkeeping.

## Derivation
Pooled PQ = sum of matched IoU / (TP + FP/2 + FN/2). Keeping a candidate adds 1/2 to the denominator for every annotator
view (matched: TP + 1 and FN - 1; unmatched: FP + 1) and its matched IoU to the numerator. Hence keeping candidate c raises
PQ if and only if

    u(c) > PQ / 2,   u(c) = mean over views of (IoU if IoU > 0.5 else 0) = y(c) * q(c),

with y = share of annotator views that match the candidate and q = E[IoU | match]. The optimal global rule is a single
threshold tau* = PQ*/2, which can be found without tuning by Dinkelbach iteration tau <- PQ(tau)/2.

## Checks
- Oracle (true u): tau 0.20 / 0.24 / **0.26** / 0.28 / 0.30 -> PQ 0.5154 / 0.5173 / **0.5175** / 0.5172 / 0.5167.
  The best tau is 0.5175 / 2.
- E[IoU | match] = 0.6725 (sd 0.085); corr(y, u) = 0.989 -> predicting y (share of annotators) and multiplying by a constant
  is equivalent to predicting u; a GBM trained on u directly is slightly worse (0.4422 vs 0.4429).
- Dinkelbach with the learned score converges in 2 iterations (tau = 0.221, PQ 0.4419, within 0.001 of the tuned best).
- Our tuned threshold y >= 0.40 corresponds to u = 0.40 x 0.67 = 0.27, i.e. about PQ/2.
