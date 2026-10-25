# retrieval.py — gloss retrieval over the full RecognitionTasks set (SignPlug Sec 5.2 /
# Table 3 at scale). Gallery = RGB_Videos embeddings (real signer). Queries =
# Generated_Videos (pose-corrected) and Not_Corrected_videos (raw SMPLer-X). Embeddings
# are L2-normalised Uni-Sign features, so similarity = dot product = cosine.
#
# Writes to --out:
#   summary.txt / summary.csv           top-1/2/3/5/10 accuracy, MRR, mean/median rank, same-label cosine
#   retrieval_<cond>.csv                per query: rank of true gloss, true-gloss cosine, top-10 glosses + scores
#   table3_style_<cond>.csv             Query | Top-1 | Score | Top-2 | Score  (paper Table 3 layout)
#   paired_rank_change.csv              per gloss: raw rank vs corrected rank
import argparse, csv, os
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--emb-dir', required=True, help='folder with RGB_Videos.pt, Generated_Videos.pt, Not_Corrected_videos.pt')
ap.add_argument('--out', required=True)
ap.add_argument('--topk', type=int, nargs='+', default=[1, 2, 3, 5, 10])
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
K = max(args.topk)


def load(name):
    d = torch.load(os.path.join(args.emb_dir, f'{name}.pt'), map_location='cpu')
    return {m['id']: e.numpy() for m, e in zip(d['metadata'], d['embeddings'])}


gallery = load('RGB_Videos')
conds = {'corrected': load('Generated_Videos'), 'not_corrected': load('Not_Corrected_videos')}

# Evaluate on glosses embedded in all three sets, so both conditions share one gallery and query list.
common = sorted(set(gallery).intersection(*[set(q) for q in conds.values()]))
lost = {n: sorted(set(s) - set(common)) for n, s in [('RGB', gallery)] + list(conds.items())}
G = np.stack([gallery[g] for g in common])
idx = {g: i for i, g in enumerate(common)}

summary_rows, ranks = [], {}
for cond, qset in conds.items():
    Q = np.stack([qset[g] for g in common])
    S = Q @ G.T                                         # [n_query, n_gallery]
    order = np.argsort(-S, axis=1)
    true_score = S[np.arange(len(common)), np.arange(len(common))]
    rank = (S > true_score[:, None]).sum(1) + 1         # 1 = true gloss retrieved first
    ranks[cond] = rank

    with open(os.path.join(args.out, f'retrieval_{cond}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['query', 'rank_of_true', 'true_cosine'] +
                   [c for k in range(1, K + 1) for c in (f'top{k}', f'top{k}_cosine')])
        for i, g in enumerate(common):
            w.writerow([g, int(rank[i]), f'{true_score[i]:.4f}'] +
                       [c for j in order[i, :K] for c in (common[j], f'{S[i, j]:.4f}')])
    with open(os.path.join(args.out, f'table3_style_{cond}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Query', 'Top-1', 'Score', 'Top-2', 'Score', 'Top-1 correct'])
        for i, g in enumerate(common):
            a, b = order[i, 0], order[i, 1]
            w.writerow([g, common[a], f'{S[i, a]:.4f}', common[b], f'{S[i, b]:.4f}', common[a] == g])

    row = {'condition': cond, 'n_queries': len(common), 'gallery_size': len(common)}
    for k in args.topk:
        row[f'top{k}_acc'] = float((rank <= k).mean())
    row.update(MRR=float((1.0 / rank).mean()), mean_rank=float(rank.mean()),
               median_rank=float(np.median(rank)), mean_true_cosine=float(true_score.mean()),
               chance_top1=1.0 / len(common))
    summary_rows.append(row)

with open(os.path.join(args.out, 'summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
    w.writeheader(); w.writerows(summary_rows)

rc, rn = ranks['corrected'], ranks['not_corrected']
with open(os.path.join(args.out, 'paired_rank_change.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['gloss', 'rank_not_corrected', 'rank_corrected', 'improvement'])
    for i in np.argsort(-(rn - rc)):
        w.writerow([common[i], int(rn[i]), int(rc[i]), int(rn[i] - rc[i])])

better, worse = int((rc < rn).sum()), int((rc > rn).sum())
try:
    from scipy.stats import wilcoxon
    p = wilcoxon(rc, rn, alternative='less').pvalue
    test = f'Wilcoxon signed-rank (corrected rank < not-corrected rank): p = {p:.3g}'
except Exception as e:   # scipy missing or all ties
    test = f'Wilcoxon test unavailable ({e})'

with open(os.path.join(args.out, 'summary.txt'), 'w') as f:
    f.write(f'Gallery: RGB_Videos ({len(common)} glosses). Queries: same {len(common)} glosses per condition.\n')
    f.write(f'Chance top-1 = 1/{len(common)} = {1 / len(common):.4f}\n')
    for n, l in lost.items():
        if l:
            f.write(f'Not in all three embedding sets, excluded from {n}: {l}\n')
    f.write('\n' + f"{'condition':15s}" + ''.join(f'{"top" + str(k):>8s}' for k in args.topk)
            + f"{'MRR':>8s}{'meanR':>8s}{'medR':>7s}{'cos(true)':>11s}\n")
    for r in summary_rows:
        f.write(f"{r['condition']:15s}" + ''.join(f"{100 * r[f'top{k}_acc']:7.2f}%" for k in args.topk)
                + f"{r['MRR']:8.4f}{r['mean_rank']:8.1f}{r['median_rank']:7.0f}{r['mean_true_cosine']:11.4f}\n")
    f.write(f'\nPer gloss, rank of the true gloss: corrected better for {better}, worse for {worse}, '
            f'tied for {len(common) - better - worse}.\n{test}\n')
print(open(os.path.join(args.out, 'summary.txt')).read())
