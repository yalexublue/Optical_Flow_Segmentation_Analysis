import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.metrics import pairwise_distances
from scipy import linalg
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans, SpectralClustering
from scipy import sparse
import sklearn as sk

# import seaborn as sns
# sns.set_style('darkgrid', {'axes.facecolor': '.9'})
# sns.set_palette(palette='deep')
# sns_c = sns.color_palette(palette='deep')

#Step 1: Compute Graph Laplacian, result is generated from subtracting ajacency matrix from degree matrix
#Generated using nearest neighbors, can be modified to interpolation
#nn is the neighborhood, df is the data set. Doesn't have to be a affinity matrix.
'''df can be a affinity matrix, however gaussian affinity matrix is to be tested'''
def generate_graph_laplacian(df, nn):
    #Generate graph Laplacian from data.
    # Adjacency Matrix.
    connectivity = kneighbors_graph(X=df, n_neighbors=nn, mode='connectivity')
    adjacency_matrix_s = (1/2)*(connectivity + connectivity.T)
    print('adjacency matrix:')
    print(adjacency_matrix_s.todense())
    # Graph Laplacian.
    graph_laplacian_s = sparse.csgraph.laplacian(csgraph=adjacency_matrix_s, normed=False)
    graph_laplacian = graph_laplacian_s.toarray()
    print('graph laplacian:')
    print(graph_laplacian)
    return graph_laplacian

#Another method generating lablacian
"""
W = pairwise_distances(X, metric="euclidean")
#print(W)
#This step forward is converting the affinity matrix to later be clustered.
#5 here is the threshold.
#start from here if obtained the affinity matrix with distance value
vectorizer = np.vectorize(lambda x: 1 if x < 5 else 0)
W = np.vectorize(vectorizer)(W)
#print(W)
def draw_graph(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
#Randomly generate adjancency matrix.
#As a prerequisite, all nodes must can be able to connect to each other.
G = nx.random_graphs.erdos_renyi_graph(10, 0.5)
draw_graph(G)
W = nx.adjacency_matrix(G)
print ("adjacency matrix")
print(W.todense())
#Build degree matrix and Laplacian matrix
# degree matrix
#This matrix value all on the diagnal
#each number represent the sum of the adjacency matrix.
D = np.diag(np.sum(np.array(W.todense()), axis=1))
print('degree matrix:')
print(D)
# laplacian matrix
#An easy stem, subtract adjacency matrix from the degree matrix
L = D - W
print('laplacian matrix:')
print(L)
"""

def compute_spectrum_graph_laplacian(graph_laplacian):
    #Compute eigenvalues and eigenvectors and project them onto the real numbers.
    eigenvals, eigenvcts = linalg.eig(graph_laplacian)
    eigenvals = np.real(eigenvals)
    eigenvcts = np.real(eigenvcts)
    print(eigenvals)
    return eigenvals, eigenvcts

def project_and_transpose(eigenvals, eigenvcts, num_ev):
    #Select the eigenvectors corresponding to the first (sorted) num_ev eigenvalues as columns in a data frame.
    #Sort them out is so that we can locate the 0 eigen value and their corresponding vectors
    #which will be determined as our clusters
    eigenvals_sorted_indices = np.argsort(eigenvals)
    indices = eigenvals_sorted_indices[: num_ev]

    proj_df = pd.DataFrame(eigenvcts[:, indices.squeeze()])
    proj_df.columns = ['v_' + str(c) for c in proj_df.columns]
    return proj_df

def run_k_means(df, n_clusters):
    #built in K means clustering, cluster number is obtained though try and test using below commented out
    #algorithm. Clusters equal when inertial first equals 0
    k_means = KMeans(random_state=25, n_clusters=n_clusters)
    k_means.fit(df)
    cluster = k_means.predict(df)
    return cluster

"""
# implementation to search through find the correct cluster number
inertias = []
k_candidates = range(1, 6)
for k in k_candidates:
    k_means = KMeans(random_state=42, n_clusters=k)
    k_means.fit(proj_df)
    inertias.append(k_means.inertia_)
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=k_candidates, y = inertias, s=80, ax=ax)
sns.lineplot(x=k_candidates, y = inertias, alpha=0.5, ax=ax)
ax.set(title='Inertia K-Means', ylabel='inertia', xlabel='k');    
"""

#this function will return "tag" it will be the same size as the data set which generated
#the affinity matrix, data set should have row number as total amount of data, in our case
#an array of Track class object, trajectories will do fine.
def spectral_clustering(df, n_neighbors, n_clusters):
    graph_laplacian = generate_graph_laplacian(df, n_neighbors)
    eigenvals, eigenvcts = compute_spectrum_graph_laplacian(graph_laplacian)
    proj_df = project_and_transpose(eigenvals, eigenvcts, n_clusters)
    cluster = run_k_means(proj_df, proj_df.columns.size)
    print (len(cluster))
    print (cluster)
    #this will add an extra column in the end of the each coord x and y displaying its
    #belonging cluster
    return cluster

#tag data by cluster result
def data_tagging(data_set, cluster):
    count = 0
    for i in range(0, len(X)):
        data_set[i].label = cluster[count]
        print(cluster[count])
        count += 1