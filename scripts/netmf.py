#!/usr/bin/python

from sys import argv
import emBench
graph_name = argv[1]

g = emBench.load('data/%s_clg.pickle' % graph_name)
dw = emBench.scorers.NetMFScorer()
dw.fit_to_graph(g)
dw.save('data/%s_nmf.pickle' % graph_name)
