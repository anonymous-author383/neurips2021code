#!/usr/bin/python

from sys import argv
import emBench
graph_name = argv[1]

g = emBench.load('data/%s_clg.pickle' % graph_name)
dw = emBench.scorers.HandTunedScorer()
dw.fit_to_graph(g, workers=24)
dw.save('data/%s_ht.pickle' % graph_name)
