from aliases import *
import similarity as sim

class Dendrogram:
    """
    Dendrogram is a binary tree which represents hierarchical clustering
    ( tree of clusters )
    The root of dendrogram is whole dataset
    """
    def __init__( self, cluster: Cluster ) -> None:
        self.cluster = cluster
        self.left: "Dendrogram" | None = None
        self.right: "Dendrogram" | None = None
        self.parent: "Dendrogram" | None = None

def merge_clusters( c1: Cluster, c2: Cluster ) -> Cluster:
    """
    Merges two clusters into one
    """
    return c1 + c2


def most_similar( clusters: list[ Cluster ] ) -> tuple[ Cluster, Cluster ]:
    """
    Finds pair of clusters which are the most similar
    """
    
    a, b = None, None
    max_sim = -1

    for c1 in clusters:
        for c2 in clusters:
            if c1 == c2:
                continue

            curr_sim = sim.cluster_similarity( c1, c2 )
            if curr_sim >= max_sim:
                max_sim = curr_sim
                a, b = c1, c2

    assert a is not None
    assert b is not None
    return a, b


def find_dendrogram( dendrograms: list[ Dendrogram ], cluster: Cluster ) -> Dendrogram:

    for d in dendrograms:
        if cluster == d.cluster:
            return d

    assert False

def hac( data: Dataset ) -> Dendrogram:
    """
    Hierarchical algomerative clustering

    returns - Dendrogram of clustering
    """

    clusters = [ [ row.tolist() ] for _, row in data.iterrows() ]
    dendrograms = [ Dendrogram( cluster ) for cluster in clusters ]

    while len( clusters ) != 1:
        c1, c2 = most_similar( clusters )
        clusters.remove( c1 )
        clusters.remove( c2 )
        new = merge_clusters( c1, c2 )
        clusters.append( new )

        ## --- Attaching new dendrogram
        d1 = find_dendrogram( dendrograms, c1 )
        d2 = find_dendrogram( dendrograms, c2 )
        d = Dendrogram( new )
        d.left = d1
        d.right = d2
        d1.parent = d
        d2.parent = d
        ## ---
        dendrograms.append( d )

    return dendrograms[ -1 ]
