#!/usr/bin/python

from sys import argv
import emBench
graph_name = argv[1]

g = emBench.load('data/%s_clg.pickle' % graph_name)
n2v = emBench.scorers.Node2VecScorer()
n2v.fit_to_graph(g)
n2v.save('data/%s_n2v.pickle' % graph_name)
