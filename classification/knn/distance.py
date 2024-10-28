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

