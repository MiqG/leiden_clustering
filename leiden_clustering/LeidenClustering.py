import numpy as np
from sklearn.decomposition import PCA
from umap.umap_ import nearest_neighbors
import leidenalg
from scanpy.neighbors import _compute_connectivities_umap
from scanpy._utils import get_igraph_from_adjacency


class LeidenClustering:
    """
    Leiden Clustering
    
    Infers nearest neighbors graph from data matrix and uses the Leiden algorithm
    to cluster the observations into clusters or communities.
    Specifically, (i) compress information with PCA, (ii) compute nearest
    neighbors graph with UMAP, (iii) cluster graph with leiden algorithm.
    
    This is a class wrapper based on https://github.com/theislab/scanpy/blob/c488909a54e9ab1462186cca35b537426e4630db/scanpy/tools/_leiden.py.
    
    Parameters
    ----------
    pca_kws : dict, default={"n_components":10}
        Parameters to control PCA step using `sklearn.decomposition.PCA`.
        
    nn_kws :  dict, default={"n_neighbors": 30, "metric": "cosine", 
    "metric_kwds": {}, "angular": False, "random_state": np.random}
        Parameters to control generation of nearest neighbors graph using 
        `umap.umap_.nearest_neighbors`.
    
    partition_type : type of `class`, default=leidenalg.RBConfigurationVertexPartition
        The type of partition to use for optimization of the Leiden algorithm.
        
    leiden_kws : dict, default={"n_iterations": -1, "seed": 0}
        Parameters to control Leiden algorithm using `leidenalg.find_partition`.
    
    Attributes
    ----------
    labels_ : ndarray of shape (n_samples)
        Cluster labels for each data point.
    
    Examples
    --------
    from leiden_clustering import LeidenClustering
    import numpy as np
    X = np.random.randn(100,10)
    clustering = LeidenClustering()
    clustering.fit(X)
    clustering.labels_
    """
    def __init__(
        self,
        pca_kws={"n_components":10},
        nn_kws={
            "n_neighbors": 30,
            "metric": "cosine",
            "metric_kwds": {},
            "angular": False,
            "random_state": np.random,
        },
        partition_type=leidenalg.RBConfigurationVertexPartition,
        leiden_kws={"n_iterations": -1, "seed": 0},
    ):
        self.pca_kws = pca_kws
        self.nn_kws = nn_kws
        self.partition_type = partition_type
        self.leiden_kws = leiden_kws
        
    def fit(self, X):
        # compress information with PCA
        pca = PCA(**self.pca_kws)
        pcs = pca.fit_transform(X)
        
        # compute nearest neighbors with UMAP
        knn_indices, knn_dists, forest = nearest_neighbors(pcs, **self.nn_kws)

        # compute connectivities
        distances, connectivities = _compute_connectivities_umap(
            knn_indices, knn_dists, pcs.shape[0], self.nn_kws["n_neighbors"]
        )

        # use connectivites as adjacency matrix to get igraph
        G = get_igraph_from_adjacency(connectivities, directed=True)

        # run leiden on graph
        self.leiden_kws["weights"] = np.array(G.es["weight"]).astype(np.float64)

        partition = leidenalg.find_partition(G, self.partition_type, **self.leiden_kws)
        labels = np.array(partition.membership)
        
        self.labels_ = labels
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
