# Network Analysis of Mangoes

This reository contains the code for evaluation of word embedding library in python named Mangoes, provided by https://gitlab.inria.fr/magnet/mangoes.

Requirements:
Python 3.5
mangoes
annoy
sklearn
scipy
numpy
random
networkx
community
matplotlib

graph evaluation.ipynb contains the code for graph contruction based on word embedding matrix. We implemented 3 types of graph: knn graph, approximate knn graph and relative neighbour graph. Also this file will help evaluate the graph properties including degree distribution, diameter, clustering coefficient and community properties. Based on relative neighbour graph, you can also visualize the rng tree with certain root word in this file.

comparison.ipynb In this python notebook, we present the comparison of pretrained word2vec and mangoes method with word2vec and GloVe. The word embedding size is 20 and the window size is 5. We build knn graph to compare degree distribution, diameter, clustering coefficient and community.


