# neurips2021code
This repository contains code for a neurips2021 submission.

##INSTALLATION:
```
pip install emBench/
```
Requires the numpy, matplotlib, sklearn, numba and gensim packages


##DATASETS:
SBMs are generated with networkx, and dblp and amazon are available at https://snap.stanford.edu/data/

##USAGE:
All datasets should be in emBench.graphs.CommLabeledGraph objects, and then saved with the save method into pickle objects. 
The scripts/ directory contains scripts to generate the embeddings, compute hadamard product models and plot reliability plots.
