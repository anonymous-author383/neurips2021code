#!/usr/bin/python3
"""Usage:
    reliability_curves.py DATASETPATH PLOTTITLE [EMBEDDINGPATH EMBEDDINGNAME COLOR]*
"""
import numpy as np
from sys import argv
import emBench
import pickle as pkl

import matplotlib.pyplot as plt
k = 10

def histofy(ps, num_bins):
    bins = np.zeros(num_bins, dtype=np.int32)
    for p in ps:
        bins[int( np.floor( p * (num_bins - 1)) )] += 1
    return bins


graph_name = argv[1]

g = emBench.load(graph_name)
targets = argv[3:]
fig, ax = plt.subplots()
title = argv[2]
num_rows = 1000
pairs = [emBench.utils.sample_from_comm(g, 1) for _ in range(num_rows)]
num_bins = 8
labels = [g.label_pairs(p) for p in pairs]
for j, target in enumerate(targets):
    print(target)
    scorer_name, label, color = target.split(' ')
    scorer = emBench.load(scorer_name)
    scores = [scorer.score_pairs(p) for p in pairs]
    x = [np.argsort(-s, axis=0) for s in scores]
    ps = np.array([emBench.analysis._top_ps(scores[l], labels[l], x[l], 10) for l in range(num_rows)])

    f = open(scorer_name.split('.')[0] + '_sample.pickle', 'wb')
    pkl.dump(ps, f)
    f.close()

    bins = histofy(ps, num_bins)
    ax.plot(np.arange(num_bins), bins, label=label, color=color)


ax.set_xticks(np.arange(num_bins))
ax.set_xticklabels(["%0.2f" % ((foo + 1) / num_bins) for foo in range(num_bins)])
ax.legend(fontsize='x-large')
ax.set_xlabel('Precision@10', fontsize='x-large')
ax.set_ylabel('Fraction of nodes', fontsize='x-large')
ax.set_title('Precision@10 reliability curve on %s' % (title), fontsize='xx-large')
#handles, labels = axs[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper left')
plt.tight_layout()
fig.savefig('figures/%s_curves.png' % (title))
