import gain
import pandas as pd

GOAL = "Tenis"
CHART = pd.read_excel( "tenis.xlsx" )

Attribute = str


class Node:
    def __init__( self, is_leaf: bool, attr: Attribute ) -> None:
        self.is_leaf = is_leaf
        self.attr = attr  # if is_leaf, this represent result class
        self.parent: "Node" | None = None
        self.children: dict[ str, "Node" ] = {} # { atribute value -> child node }

    def draw( self, indent: int=0, attr_val: str="" ) -> None:

        print( indent * " ", end="" )

        attr = self.parent.attr if self.parent is not None else ""
        attr_draw = f"( {attr}={attr_val} )" if attr != "" else ""
        val = self.attr + "?" if not self.is_leaf else self.attr
        print( val, attr_draw )

        for attr_val, child in self.children.items():
            child.draw( indent + 4, attr_val )


## !! CURENTLY NOT WORKING WITH COUNTINOUS-VALUED ATTRIBUTES !!

class DecisionTree:
    def __init__( self, root: Node ) -> None:
        self.root = root

    def get_root( self ) -> Node:
        return self.root

    def draw( self ) -> None:
        self.root.draw()


def best_attr( data: pd.DataFrame ) -> Attribute:
    """
    Function to find attribute with the highest information gain from data
    """
    gains = [ ( gain.gain( data, attr, GOAL ), attr ) for attr in data if attr != GOAL ]
    return max( gains )[ 1 ]


def split_by( data: pd.DataFrame, attr: Attribute ) -> list[ tuple[ str, pd.DataFrame ] ]:
    """
    Function to split data on given attr ( i.e. creates new dataframes for every value of given attr )
    """
    vals = gain.get_distr( data, attr )
    return [ ( val, data.loc[ data[ attr ] == val ] ) for val in vals ]


def decision_tree( data: pd.DataFrame ) -> DecisionTree:
    """
    Returns decision tree for given data
    """
    root = decision_tree_rec( data, "", set() )
    return DecisionTree( root )


def decision_tree_rec( data: pd.DataFrame, attr_val: str, seen_attrs: set[ str ] ) -> Node:
    """
    Function which returns a partial decision tree node for given data
    """
    classes = data[ GOAL ].unique()

    ## If all examples are from single class
    if len( classes ) == 1:
        return Node( is_leaf=True, attr=classes[ 0 ] )

    attr = best_attr( data )

    ## If we already splitted on this attr -> there will never be exactly one class
    ## so we create leaf node with most common value from examples
    if attr in seen_attrs:
        most_common = data[ GOAL ].mode()   ## Can be more values
        return Node( is_leaf=True, attr=most_common[0] )

    seen_attrs.add( attr )

    new = Node( is_leaf=False, attr=attr )

    for attr_val, partition in split_by( data, attr ):
        child = decision_tree_rec( partition, attr_val, seen_attrs )
        child.parent = new
        new.children[ attr_val ] = child

    return new

tree = decision_tree( CHART )
tree.draw()
