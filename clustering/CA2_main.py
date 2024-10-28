import clustering_resources as cr
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import datetime

SEED = 2407741
cr.RandomNumGen.set_seed(SEED)
rng = cr.RandomNumGen.get_gen()

# --- Helpers
def clr_transform(sample, eps=1e-9):
    sample[sample == 0] = eps
    geometric_mean = np.exp(np.mean(np.log(sample)))
    return np.log(sample / geometric_mean)

def plot_clustering(data: NDArray, clustering: NDArray, name: str, x: str, y:str) -> None:

    plt.title(name)
    plt.xlabel(x)
    plt.ylabel(y)

    plt.scatter(data[:, 0], data[:, 1], c=clustering)
    plt.show()
# ---

# --- Task 1
class KMeans:
    def __init__(self, k: int) -> None:
        assert k > 0
        self.data = None
        self.k = k

    def centroid(self, clustering: NDArray, c: int) -> NDArray:
        points = self.data[ clustering == c,: ]
        assert len(points) != 0
        return np.mean(points, axis=0)

    def recalculate_centroids(self, clustering: NDArray) -> NDArray:
        return np.array( [ self.centroid( clustering, c ) for c in range(self.k) ] )

    def fit(self, data: NDArray) -> NDArray:
        assert len(data) >= self.k
        self.data = data

        indices = rng.choice(np.arange(len(data)), size=self.k, replace=False)
        centroids = self.data[ indices, : ]
        clustering = np.array([2])
        new_clustering = np.array([1]) # will be overwritten in the beggining of the loop

        while not np.array_equal(clustering, new_clustering):
            clustering = new_clustering
            distance_matrix = cr.L2_distance(self.data, centroids)
            new_clustering = np.argmin(distance_matrix, axis=1)
            centroids = self.recalculate_centroids(new_clustering)

        return clustering
# ---

# --- Task 2
class BisectingKMeans():
    def __init__(self, k: int) -> None:
        self.k = k
        self.data = None
        self.cohesiveness = {}

    def next_to_split(self, clustering: NDArray) -> int:
        """
        Selecting the least cohesive cluster
        """
        clusters = np.unique(clustering)

        for c in clusters:
            if c not in self.cohesiveness:
                mask = (clustering == c)
                subm = self.dm[mask][:, mask]
                self.cohesiveness[c] = np.max(subm)

        return max(self.cohesiveness, key=self.cohesiveness.get)

    def fit(self, data: NDArray) -> NDArray:
        assert len(data) >= self.k
        self.data = data

        self.dm = cr.L2_distance(self.data, self.data)

        km = KMeans(k=2)
        clusters = np.array( [ 0 for _ in range(len(self.data)) ] )

        while len( np.unique(clusters) ) != self.k:
            i = self.next_to_split(clusters)
            del self.cohesiveness[i]
            splitted = km.fit(data[clusters == i, :])
            splitted += np.max(clusters) + 1 # shifting to avoid conflicts
            clusters[ clusters == i ] = splitted

        return np.unique(clusters, return_inverse=True)[1] # transform cluster numbering
# ---

# --- Task 3
class Agglomerative():
    def __init__(self, linkage_type: str) -> None:
        assert linkage_type in ["complete", "average", "single"]
        self.linkage_type = linkage_type
        self.data = None

    def linkage(self,
                elems: dict[int, list[int]],
                dist_matrix: NDArray,
                i: int,
                j: int,
                k: int) -> float:
        
        ik = dist_matrix[i, k]
        jk = dist_matrix[j, k]

        if np.isinf(ik) or np.isinf(jk):
            return np.inf

        if self.linkage_type == "single":
            return np.min( [ik, jk] )
        elif self.linkage_type == "complete":
            return np.max( [ik, jk] )
        else:
            ni = len( elems[i] )
            nj = len( elems[j] )
            return ( ni * ik + nj * jk ) / (ni + nj)

    def closest_pair(self, dist_matrix: NDArray) -> tuple[int, int]:
        return np.unravel_index(np.argmin( dist_matrix ), dist_matrix.shape)

    def fit_dend(self, data: NDArray) -> tuple[NDArray, dict[int, list[int]]]:

        self.data = data

        # each point in separate cluster
        elements = { i: [i] for i in range( len(data) ) }
        dist_matrix = cr.L2_distance(self.data, self.data)
        np.fill_diagonal(dist_matrix, np.inf) # we do not want to pick the cluster to merge with itself
        dend = []

        while len(dend) < len(data) - 1:

            i, j = self.closest_pair(dist_matrix)

            # --- Merging clusters i and j
            dend.append( [i, j, dist_matrix[i][j], len(elements[i]) + len(elements[j]) ] )
            elements[ len(elements) ] = elements[i] + elements[j]
            # ---

            # --- computing distances for the cluster merged from i and j
            new_dists = np.array( [ self.linkage( elements, dist_matrix, i, j, k ) for k in range( dist_matrix.shape[0] ) ] )
            dist_matrix = np.vstack( [dist_matrix, new_dists] )

            new_dists = np.append(new_dists, np.inf)
            h_new_dists = new_dists.reshape(-1, 1)
            dist_matrix = np.hstack( [dist_matrix, h_new_dists] )
            dist_matrix[ [i, j], : ] = np.inf
            dist_matrix[ :, [i, j] ] = np.inf

        return elements, dend

    def fit(self, data: NDArray, k: int) -> NDArray:
        elems, dend = self.fit_dend(data)
        return self.truncate(elems, dend, k)

    def truncate(self,
                 elements: dict[int, list[int]],
                 dend: NDArray,
                 n_clusters: int) -> NDArray:

        assert self.data is not None

        n = len(self.data)
        clusters = np.ones( n )
        handled = set()

        c = 0
        for k, (i, j, _, _) in enumerate( dend[-(n_clusters - 1):] ):

            if i not in handled:
                clusters[ elements[i] ] = c
                c += 1

            if j not in handled:
                clusters[ elements[j] ] = c
                c += 1

            handled.add( n + n - 1 - n_clusters + 1 + k ) # id of this cluster

        return clusters.astype(int)
# ---

# --- Task 4
def analyze(data, head, reference, nc) -> None:

    dist_matrix = cr.L2_distance(data, data)

    # --- k-means
    km = KMeans(k=nc)
    tic = datetime.datetime.now()
    clusters = km.fit(data)
    tac = datetime.datetime.now()
    print("Running time: %s" % (tac-tic))

    print("K-means")
    # plot_clustering(data[ :, [0, 1] ], clusters, "K-means", head[0], head[1])

    print("---")
    print("SSDC: ", cr.ss_distances_centroids(data, clusters))
    print("DR: ", cr.intra_inter_distance_ratio(dist_matrix, clusters))
    print("Silhouette: ", cr.clust_silhouette(dist_matrix, clusters)[0])

    vals, cm = cr.get_clust_CM_vals(clusters, reference)
    print("contingency matrix:\n", cm) ## what do the rows and the columns respectively represent?
    print(", ".join(["%s=%.4f" % (k,v) for (k,v) in vals.items()]))

    # --- bisecting k-means
    bkm = BisectingKMeans(k=nc)
    tic = datetime.datetime.now()
    clusters = bkm.fit(data)
    tac = datetime.datetime.now()
    print("Running time: %s" % (tac-tic))

    print("Bisecting k-means")
    # plot_clustering(data[ :, [0, 1] ], clusters, "Bisecting k-means", head[0], head[1])

    print("---")
    print("SSDC: ", cr.ss_distances_centroids(data, clusters))
    print("DR: ", cr.intra_inter_distance_ratio(dist_matrix, clusters))
    print("Silhouette: ", cr.clust_silhouette(dist_matrix, clusters)[0])

    vals, cm = cr.get_clust_CM_vals(clusters, reference)
    print("contingency matrix:\n", cm) ## what do the rows and the columns respectively represent?
    print(", ".join(["%s=%.4f" % (k,v) for (k,v) in vals.items()]))

    # --- Agglomerative
    for l in ["complete", "single", "average"]:
        agg = Agglomerative(linkage_type=l)

        tic = datetime.datetime.now()
        clusters = agg.fit(data, nc)
        tac = datetime.datetime.now()

        print("---")
        print(f"Agglomerative {l}")
        print("Running time: %s" % (tac-tic))

        # plot_clustering(data[ :, [0, 1] ], clusters, f"Agglomerative {l}", head[0], head[1])

        print("SSDC: ", cr.ss_distances_centroids(data, clusters))
        print("DR: ", cr.intra_inter_distance_ratio(dist_matrix, clusters))
        print("Silhouette: ", cr.clust_silhouette(dist_matrix, clusters)[0])

        vals, cm = cr.get_clust_CM_vals(clusters, reference)
        print("contingency matrix:\n", cm) ## what do the rows and the columns respectively represent?
        print(", ".join(["%s=%.4f" % (k,v) for (k,v) in vals.items()]))

        print("---")
    # ---

# ---

if __name__ == "__main__":


    # BEANS EXERCISE
    dataset, head, classes = cr.load_csv("data.txt")
    data = dataset[:, [1, 2]]
    reference = np.array([ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2 ])

    analyze(data, head, reference, 3)


    # BEANS NOTEBOOK
    dataset, head, classes = cr.load_csv("Dry_Bean_Dataset_small.csv", last_column_str=True)

    var_map = dict([(vn, vi) for (vi, vn) in enumerate(head)])
    vnames6v = ['MajorAxisLength', 'MinorAxisLength', 'Eccentricity', 'roundness', 'Extent', 'ShapeFactor2']

    ref = dataset[:, var_map["class"]].astype(int)

    X = dataset[:,[var_map[v] for v in vnames6v]]
    X = cr.normalize_dataset(X)
    analyze(X, head, ref, 8)

    # GLASS NOTEBOOK
    dataset, head, classes = cr.load_csv("glass.data")
    cols = ['na', 'mg', 'al', 'si', 'k', 'ca', 'ba', 'fe']
    var_map = dict([(vn, vi) for (vi, vn) in enumerate(head)])
    ref = dataset[:, var_map["type"]].astype(int)
    X = dataset[:,[var_map[v] for v in cols]]
    X = np.apply_along_axis(clr_transform, axis=1, arr=X)
    analyze(X, cols, ref, 7)
