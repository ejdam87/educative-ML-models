from math import log2
import pandas as pd
from collections import Counter

Attribute = str

def get_entropy( probs: list[ float ] ) -> float:
    """
    Function to calculate entropy of given list of probabilities
    """
    res = 0.0
    for p in probs:
        res += -p * log2( p )

    return res

def get_distr( data: pd.DataFrame, attr: Attribute ) -> dict[ str, int ]:
    """
    Returns distribution of values of given attr { value of attribute -> count of occurances }
    """
    return Counter( data[ attr ] )


def get_partial_distr( data: pd.DataFrame,
                       attr: Attribute,
                       val: str,
                       goal: str ) -> dict[ str, int ]:
    """
    Returns distribution of rows with attribute value val
    """
    partial = data.loc[ data[ attr ] == val ]
    return get_distr( partial, goal )


def get_probs( distr: dict[ str, int ] ) -> list[ float ]:
    """
    Returns probability distribution of given distribution
    """
    total = sum( distr.values() )
    return [ p / total for p in distr.values() ]


def entropy( data: pd.DataFrame, goal: str ) -> float:
    """
    Returns entropy for given dataset with given goal attribute
    """
    distr = get_distr( data, goal )
    return get_entropy( get_probs( distr ) )


def gain( data: pd.DataFrame, attr: Attribute, goal: str ) -> float:
    """
    Calculates information gain of given dataset on given attribute with given goal attribute
    """
    res = entropy( data, goal )
    distr = get_distr( data, attr )
    total = sum( distr.values() )

    for val, count in distr.items():
        partial_distr = get_partial_distr( data, attr, val, goal )
        res -= ( count / total ) * get_entropy( get_probs( partial_distr ) )

    return res
