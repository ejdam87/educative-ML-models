import pandas as pd

Attribute = str

class Node:
    def __init__( self, is_leaf: bool, val: Attribute | int ) -> None:
        self.is_leaf = is_leaf
        self.val = val  ## result if is_leaf, split attr otherwise
        
        self.children: dict[ str, "Node" ] = {}


class RegressionTree:
    def __init__( self, root: Node ) -> None:
        self.root = root


def split_by( data: pd.DataFrame, attr: Attribute ) -> list[ tuple[ str, pd.DataFrame ] ]:
    """
    Function to split data on given attr ( i.e. creates new dataframes for every value of given attr )
    """
    vals = gain.get_distr( data, attr )
    return [ ( val, data.loc[ data[ attr ] == val ] ) for val in vals ]
