import networkx as nx
from community import community_louvain
import matplotlib.pyplot as plt
from random import random
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import lil_matrix
from construction import Construction
import numpy as np
import time


class Property:
    
    def __init__(self, graph):
        
        global G
        G = graph
    
    def visualize(self):
        plt.figure(figsize=(20,15))
        options = {'node_color': 'red',
                   'node_size': 10,
                   'width': 0.5,}
        nx.draw(G, **options)
        plt.show()
        
    def rn_tree(self, nodes, label, root = 0, k_neig = 50):
        
        def find_min(list_cand, tmp, c, a):
            '''
            Find the min distance between node c and a given list + node a, return the node with min distance

            parameters:
            list_cand: list of candidate node
            tmp: distance matrix

            '''
            min_v = tmp[a][c]
            min_d = a
            for i in list_cand:
                if tmp[i][c] > min_v:
                    min_v = tmp[i][c]
                    min_d = i
            return min_d

        def build_btw_break(k_neig, tmp):
            '''
            Build a Between Two Points Dict for large dataset, the complexity is O(k^3).
            We intend to find the node of which the btw is empty, so the btw is not complete.

            parameters: 
            k_neig: the dimension
            tmp: distance matrix
            '''
            btw ={}
            begin = time.time()
            for i in range(k_neig):
            #for i in range(0,15):
                btw[i] = {}
                for j in range(k_neig):
                    btw[i][j] = []
                    if i == j:
                        continue
                    dis = tmp[i,j]
                    for k in range(k_neig):
                        if tmp[i][k] > dis and tmp[j][k] > dis:
                            btw[i][j].append(k)
                            break
            print ('time: ' + str(time.time()-begin))
            return btw

        def build_btw(k_neig, tmp):
            '''
            Build a Between Two Points Dict for small dataset, the complexity is O(k^3).
            We intend to find the node of which the btw is not empty, so the btw is complete.

            parameters: 
            k_neig: the dimension
            tmp: distance matrix
            '''
            btw ={}
            begin = time.time()
            for i in range(k_neig):
            #for i in range(0,15):
                btw[i] = {}
                for j in range(k_neig):
                    btw[i][j] = []
                    if i == j:
                        continue
                    dis = tmp[i,j]
                    for k in range(k_neig):
                        if tmp[i][k] > dis and tmp[j][k] > dis:
                            btw[i][j].append(k)
#             print (time.time()-begin)
            return btw

        def build_sparse_m_whole(rows, btw):
            '''
            Build a sparse matrix between nodes of which the btw is empty

            parameters:
            rows: the dimension
            btw: Between Two Points Dict
            '''
            sps_acc = sps.lil_matrix((rows, rows))
            for i in range(0,rows):
                for j in range(rows):
                    if btw[i][j] == [] and i!=j:
                        sps_acc[i,j] = 1
                        sps_acc[j,i] = 1
            return sps_acc

        def build_sparse_m(rows, btw, tmp):
            '''
            Build a sparse matrix between c and btw[c][b] where distance of b and c is minimum

            parameters:
            root: the root node, build a graph with k-nearest neighbors of root
            rows: the dimension
            tmp: distance matrix
            btw: Between Two Points Dict
            '''
            sps_acc = lil_matrix((rows, rows))
            a = 0
            for c in range(0,rows):
                b = find_min(btw[a][c], tmp, c, a)
                #print (b)
                sps_acc[c,b] = 1
                sps_acc[b,c] = 1
            return sps_acc

        def find_k_neighbor(root, k_neig, cos_dis):
            '''
            Find the k-nearest neighbors of root and return the distance matrix

            parameters:
            cos_dis: cosine similarity matrix
            '''
            ind = np.argsort(-cos_dis[root])
            ind = ind[0:k_neig]
            ind = np.append([root], ind)
            k_neig = k_neig + 1
            tmp = cos_dis[np.ix_(list(ind),list(ind))]
            return tmp, k_neig, ind


        cos_dis = cosine_similarity(nodes,nodes) - np.eye(nodes.shape[0])
        tmp, k_neig, ind  = find_k_neighbor(root, k_neig, cos_dis)

            
        btw_k = build_btw(k_neig, tmp)

        # all nodes
        sps_acc = build_sparse_m(k_neig, btw_k, tmp)
        #print (sps_acc)      
        G = nx.from_scipy_sparse_matrix(sps_acc)

        dict_label = {i:label[ind[i]] for i in range(len(ind))}
        pos = nx.spring_layout(G,k=0.05)
        plt.figure(1,figsize=(20,15)) 
        nx.draw_networkx(G, labels= dict_label,pos=pos, with_labels = True, node_size=15, font_size=15)
        plt.show()
        
    def degree_dist(self):
        degree_sequence=sorted(dict(nx.degree(G)).values(),reverse=True) # degree sequence

        print('Max degree:',max(degree_sequence))
        
        plt.figure(figsize=(20,6))
        plt.subplot(121)
        plt.hist(degree_sequence)
        plt.yscale("log", nonposy='clip')
        plt.grid()
        plt.title("Histogram of degree distribution")
        plt.ylabel("Histogram")
        plt.xlabel("degree")

        plt.subplot(122)
        plt.loglog(degree_sequence,'b-',marker='o')
        plt.title("Degree rank plot")
        plt.grid()
        plt.ylabel("degree")
        plt.xlabel("rank")

        # draw graph in inset
        plt.axes([0.45,0.45,0.45,0.45])
        # Gcc=sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)[0]
        # pos=nx.spring_layout(Gcc)
        plt.axis('off')
        plt.grid('on')
        #nx.draw_networkx_nodes(Gcc,pos,node_size=20)
        #nx.draw_networkx_edges(Gcc,pos,alpha=0.4)

        #plt.savefig("degree_histogram.png")
        plt.show()
        
        return degree_sequence
        
    def diameter(self):
        d = nx.diameter(G)
        print('Diameter:',d)
        return d
    
    def clustering_coef(self):
        coff = nx.clustering(G)
        print('Clustering coefficient:',np.mean(list(coff.values())))
        return np.mean(list(coff.values()))
    
    def community(self, resolution = 1):
        partition = community_louvain.best_partition(G, resolution = resolution)
        
        n_community = max(list(partition.values()))

        print('Modularity:', community_louvain.modularity(partition,G) )
        print('Number of communities:', n_community)
        
        #drawing
        size = float(len(set(partition.values())))
        pos = nx.spring_layout(G)
        count = 0
        plt.figure(figsize=(12,8))
        for com in set(partition.values()) :
            count = count + 1.
            list_nodes = [nodes for nodes in partition.keys()
                                        if partition[nodes] == com]
            nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20, node_color= [random(),random(),random()])

        nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5)
        plt.show()
        
        return partition