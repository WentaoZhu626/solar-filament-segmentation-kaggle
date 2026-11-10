# Conflict-graph and component-tree selection (not adopted)

Scripts: `stage2/experiments/conflict_graph.py` (stages build / select / crops / cnn),
`two_stage_shape_choice.py`, `component_tree_pipeline.py`. Official bookkeeping, out of fold.

## Idea
Candidate hypotheses from several generators per image: spine-merged instances (B), faint blobs (F), refined masks (R),
connected components of the ensemble map at thresholds 0.3-0.7 (T). 56,596 hypotheses with 165,644 overlap edges. Every
hypothesis gets a score u_hat; a pairwise non-overlapping set maximising sum(u_hat - tau) is chosen exactly (maximum-weight
independent set, MILP per connected component) with tau = PQ/2. Using only T gives the nested component tree (the
selection layer of a public 0.40 notebook) as a special case.

## Results
| hypotheses allowed | learned scorer (best variant) | oracle (true u) |
|---|---|---|
| B + F (plain keep rule) | 0.4410 | 0.5175 |
| T only (component tree) | 0.4395 | 0.5480 |
| B + F + T | 0.4401 | 0.5507 |
| R only | 0.4480 | - |
| all | 0.4416 | 0.5606 |

Full component-tree pipeline in our setting (tree -> u_hat scorer -> exact antichain -> same refiner): 0.4400 -> 0.4477,
against our CNN-judge selection -> refiner 0.4481 -> 0.4542 (better in all 5 folds).

Shape choice per kept filament (original / refined / threshold alternatives): always refined 0.4540, chosen by a CNN
0.4500-0.4517, oracle 0.4811.

## Conclusion
The oracle headroom is large, but learned scorers cannot rank alternatives of the same filament: their differences lie
within annotator disagreement. The simple pipeline (CNN-judge keep rule, then refine every kept filament) stays best.
