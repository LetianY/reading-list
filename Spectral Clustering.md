#### 1. Article: Spectral Clustering Foundation and Application

###### Notes:

1. Spectral clustering uses information from the e<u>igenvalues (spectrum)</u> of <u>special matrices built from the graph or the dataset</u>

2. Graph Laplacian

   - Normal Laplacian L = D - A
   - The Laplacian’s diagonal is the degree of our nodes, and the off diagonal is the negative edge weights (here weighted graph is considered)

3. Eigenvalues of Graph Laplacian

   ![img](https://miro.medium.com/max/1050/1*p2vrLlFxdJgGZxCGO5WBmA.gif)

   - The number of 0 eigenvalues corresponds to the number of connected components in the graph

   - First eigenvalue (=0 in the final graph): corresponding eigenvector will always have constant values

   - First non-zero eigenvalue: spectral gap
     --> Related to the density of the graph, larger if graph denser
     --> E.g., If this graph was densely connected (all pairs of the 10 nodes had an edge), then the spectral gap would be 10.

   - Second eigenvalue: Fiedler value --> Fiedler vector

     - The Fiedler value approximates the minimum graph cut needed to separate the graph into two connected components
     - Fiedler value = 0 if the graph is already two connected components
     - Fiedler vector gives us information about which side of the cut that node belongs (Positive / Negative)
     - Similar for the 3rd / 4th eigenvalues and so on; 

   - In general, we often look for the **first large gap between eigenvalues** in order to find the number of clusters expressed in our data: having four eigenvalues before the gap indicates that there is likely four clusters --> the previous eigenvectors indicating how to cut the graph

   - ![img](https://miro.medium.com/max/648/1*omUQ6aCQ88uK2rapwOFgvw.png)

     Clustered using K-means algorithm

4. K-Means
   - Operates on Euclidean distance
   - Assumes that the clusters are roughly spherical
   - ![img](https://miro.medium.com/max/1050/1*rzgNKF9GVAStuGhQnhQgeQ.png)
5. How to construct data points as a graph? 
   --> K-nearest neighbors graph

- treat data points as nodes in graph, edge is drawn to each node's k nearest neighbors in the original space 
- algorithm sensitive to k (but solves the clustering problem, codes attached below)

- ![img](https://miro.medium.com/max/1050/1*qmdN607THNkFEmR_yqWeOw.png)
- The nearest neighbor graph is a nice approach, but it relies on the fact that “close” points should belong in the same cluster

6. A more general approach: construct an affinity matrix

- Similar to adjacent matrix
- The value for a pair of points expresses how similar those points are to each other (=0 if dissimilar, might be 1 if identical)
- How to decide on what it means for two data points to be similar is one of the most important questions in machine learning
- Example: If you have some labeled data, you can train a classifier to predict whether two inputs are similar or not based on if they have the same label. This classifier can then be used to assign affinity to pairs of unlabeled points.

###### Python Codes:

```python
vals, vecs = np.linalg.eig(A)
# Adjacency matrix A is directly constructed from the graph
D = np.diag(A.sum(axis=1))
L = D-A
```

```python
# K-means clustering
from sklearn.cluster import KMeans

D = np.diag(A.sum(axis=1))
L = D-A
vals, vecs = np.linalg.eig(L)

# sort these based on the eigenvalues
vecs = vecs[:,np.argsort(vals)]
vals = vals[np.argsort(vals)]

# kmeans on first three vectors with nonzero eigenvalues
kmeans = KMeans(n_clusters=4)
kmeans.fit(vecs[:,1:4])
colors = kmeans.labels_

print("Clusters:", colors)
# Clusters: [2 1 1 0 0 0 3 3 2 2]
```

```python
# KNN to construct graph and do clustering, can differentiate the two circles
from sklearn.datasets import make_circles
from sklearn.neighbors import kneighbors_graph
import numpy as np

# create the data
X, labels = make_circles(n_samples=500, noise=0.1, factor=.2)

# use the nearest neighbor graph as our adjacency matrix
A = kneighbors_graph(X, n_neighbors=5).toarray()

# create the graph laplacian
D = np.diag(A.sum(axis=1))
L = D-A

# find the eigenvalues and eigenvectors
vals, vecs = np.linalg.eig(L)

# sort
vecs = vecs[:,np.argsort(vals)]
vals = vals[np.argsort(vals)]

# use Fiedler value to find best cut to separate data
clusters = vecs[:,1] > 0
```



###### Questions and Subtasks:

1. Try to implement the animation, to see how the change of the # of edges / edge-weight will affect the eigenvalues
2. Try to figure out why this holds: the number of 0 eigenvalues corresponds to the number of connected components in the graph
3. Can the step "k-means on first three vectors with nonzero eigenvalues" solve the clustering problem? How? Understand the codes.
4. Study affinity matrix and how ppl usually measure similarity



###### Reading:

https://people.csail.mit.edu/dsontag/courses/ml14/notes/Luxburg07_tutorial_spectral_clustering.pdf

http://cs-www.cs.yale.edu/homes/spielman/sgta/SpectTut.pdf

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1379882&tag=1

https://proceedings.neurips.cc/paper/2018/file/2a845d4d23b883acb632fefd814e175f-Paper.pdf

https://web.cs.elte.hu/~lovasz/eigenvals-x.pdf

https://en.wikipedia.org/wiki/Laplacian_matrix

https://en.wikipedia.org/wiki/Spectral_graph_theory

https://en.wikipedia.org/wiki/Spectral_gap

https://en.wikipedia.org/wiki/Radial_basis_function_network

https://cse.hkust.edu.hk/~dimitris/6311/L17-AGP-Zhao.pdf

https://www.sciencedirect.com/science/article/pii/S1319157820304511

https://www.youtube.com/watch?v=p_zknyfV9fY

###### Dataset:

https://snap.stanford.edu/data/ego-Facebook.html

https://renchi.ac.cn/datasets/

###### Codes:

https://github.com/dgleich/graph_eigs

http://www.yann-ollivier.org/specgraph/specgraph.php

https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html

#### 2. Manifold Learning

https://github.com/drewwilimitis/Manifold-Learning

https://scikit-learn.org/stable/modules/manifold.html

https://leovan.me/cn/2018/03/manifold-learning/

https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html

https://www.cis.upenn.edu/~cis6100/diffgeom-n.pdf

#### 3. continue learning

 https://www.sciencedirect.com/science/article/pii/S1364661320302199?ref=pdf_download&fr=RR-2&rr=75157f08eafe482e

https://paperswithcode.com/task/continual-learning

https://en.wikipedia.org/wiki/Attention_(machine_learning)

#### 4. Deep Clustering

https://arxiv.org/pdf/2206.07579.pdf

https://arxiv.org/pdf/2006.16904.pdf

https://arxiv.org/pdf/1906.06532v1.pdf

https://paperswithcode.com/paper/n2dnot-too-deep-clustering-via-clustering-the

https://paperswithcode.com/paper/structural-deep-clustering-network

https://github.com/zhoushengisnoob/DeepClustering

https://arxiv.org/pdf/2111.11821.pdf

https://arxiv.org/pdf/2009.09590.pdf

Contrastive clustering

https://web.archive.org/web/20220717202429id_/https://www.ijcai.org/proceedings/2022/0457.pdf

#### 5. K-edge Addition:

https://www.jmlr.org/papers/volume19/16-534/16-534.pdf

https://people.orie.cornell.edu/dpw/orie6334/lecture23.pdf

https://drops.dagstuhl.de/opus/volltexte/2020/13408/pdf/LIPIcs-ISAAC-2020-64.pdf

https://ieeexplore.ieee.org/document/8989355

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6529077
