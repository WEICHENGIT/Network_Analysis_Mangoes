from annoy import AnnoyIndex
from sklearn.neighbors import kneighbors_graph, BallTree
import networkx as nx
from scipy.sparse import coo_matrix, csc_matrix, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import numpy as np

class Construction:
    
    def __init__(self, node_matirx):
        
        global nodes
        nodes = node_matirx
    
    def approximate_knn(self, k=5, sigma=1, n_tree=10):

        #####################################
        ## constructing trees
        t = AnnoyIndex(nodes.shape[1])
        for i in range(nodes.shape[0]):
            t.add_item(i, nodes[i])

        t.build(n_tree) # 10 trees

        #####################################
        ## approximate knn
        nn = np.zeros([nodes.shape[0],k+1])
        distance = np.zeros([nodes.shape[0],k+1])

        for node in range(nodes.shape[0]):
            nn[node, :], distance[node, :] = t.get_nns_by_item(node, k+1, include_distances=True) # will find the 6 nearest neighbors
            distance[node, :] = np.exp(-distance[node, :]**2/(2*sigma**2))
        A = coo_matrix((distance[:,1:].flatten(), (np.repeat(range(nodes.shape[0]), k),nn[:,1:].flatten())), 
                       shape=(nodes.shape[0],nodes.shape[0]))

        G = nx.from_scipy_sparse_matrix(A)
        
        return G
    
    def knn(self, k=5, sigma=1):
        D = kneighbors_graph(nodes, k, mode='distance', metric='euclidean', include_self=False, n_jobs=-1).toarray()
        A = np.exp(-D**2/(2*sigma**2))
        A[A==1]=0
        A = csc_matrix(A)

        G = nx.from_scipy_sparse_matrix(A)
        
        return G
    
    def rng(self):
        
#     Relative neighbor graph.
#     ref: http://www.aclweb.org/anthology/D/D15/D15-1292.pdf
    
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
#             begin = time.time()
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
#             print ('time: ' + str(time.time()-begin))
            return btw



        def build_sparse_m_whole(rows, btw):
            '''
            Build a sparse matrix between nodes of which the btw is empty

            parameters:
            rows: the dimension
            btw: Between Two Points Dict
            '''
            sps_acc = lil_matrix((rows, rows))
            for i in range(0,rows):
                for j in range(rows):
                    if btw[i][j] == [] and i!=j:
                        sps_acc[i,j] = 1
                        sps_acc[j,i] = 1
            return sps_acc
        

        
        cos_dis = cosine_similarity(nodes,nodes) - np.eye(nodes.shape[0])
        btw = build_btw_break(nodes.shape[0], cos_dis)
        sps_acc = build_sparse_m_whole(nodes.shape[0], btw)
        G = nx.from_scipy_sparse_matrix(sps_acc)
        
        return G