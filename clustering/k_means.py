from aliases import *
import similarity as sim

import random

def k_random_indices( n: int, k: int ) -> list[ int ]:
    """
    Returns k random DISTINCT values from range 0-n
    """
    pool = list( range( n ) )
    res = []

    for _ in range( k ):
        i = random.choice( pool )
        pool.remove( i )
        res.append( i )

    return res

def closest_centroid( row: Sample, centroids: list[ Sample ] ) -> tuple[ int, Sample ]:

    ci, cc, cs = -1, None, -1
    for i, centroid in enumerate( centroids ):

        if centroid == row:
            return i, centroid

        current_similarity = sim.instance_similarity( row, centroid )
        if current_similarity >= cs:
            cs = current_similarity
            ci = i
            cc = centroid

    return ci, cc

def centroid( cluster: Cluster ) -> Sample:
    assert len( cluster ) != 0

    centroid = [ 0 for _ in range( len( cluster[0] ) ) ]
    for i in range( len( cluster[0] ) ):
        for sample in cluster:
            centroid[ i ] += sample[ i ]

    centroid = [ val / len( cluster ) for val in centroid ]
    return centroid


def recalculate_centroids( clusters: list[ Cluster ] ) -> list[ Sample ]:
    return [ centroid( cluster ) for cluster in clusters ]


def k_means( data: Dataset, k: int ) -> list[ Cluster ]:
    """
    Performs k-means clustering on given data

    returns k clusters of given data
    """

    data = [ row.tolist() for _, row in data.iterrows() ]
    centroids = [ data[ i ] for i in k_random_indices( len( data ), k ) ]

    new_clusters = []
    clusters = [ [] for _ in range( k ) ]
    
    while new_clusters != clusters:
        
        clusters = new_clusters
        new_clusters = [ [] for _ in range( k ) ]

        for row in data:
            closest_pos, closest_centr = closest_centroid( row, centroids )
            new_clusters[ closest_pos ].append( row )

        centroids = recalculate_centroids( new_clusters )

    return clusters

data = pd.read_excel( "example.xlsx" )
print( k_means( data, 2 ) )
