from aliases import *
import math

def euclidean( s1: Sample, s2: Sample ) -> float:
    """
    Calculates euclidean distance between two vectors
    """
    assert len( s1 ) == len( s2 )

    res = 0
    for i in range( len( s1 ) ):
        res += ( s1[ i ] - s2[ i ] ) ** 2

    return math.sqrt( res )

def instance_similarity( s1: Sample, s2: Sample ) -> float:
    return 1 / euclidean( s1, s2 )

def cluster_similarity( c1: Cluster, c2: Cluster ) -> float:
    """
    Single-link similarity - similarity of two most similar memebers
    """
    
    highest_similarity = -1

    for s1 in c1:
        for s2 in c2:
            highest_similarity = max( instance_similarity( s1, s2 ), highest_similarity )

    return highest_similarity
