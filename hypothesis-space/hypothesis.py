import pandas as pd

## Only binary classification !!

Attribute_value = str
Dataset = pd.DataFrame

GENERAL = "?"
SPECIFIC = "Ø"

LABEL = "Category"

class Hypothesis:
    def __init__( self, vals: list[ Attribute_value ] ) -> None:
        self.vals = vals    ## these values leads to positive class

    def __repr__( self ) -> str:
        return "<" + ",".join( self.vals ) + ">"

    def __getitem__( self, i: int ) -> Attribute_value:
        return self.vals[ i ]


def sample_consistent( h: Hypothesis, sample: pd.Series ) -> bool:
    
    label = sample[ LABEL ]

    sample = sample.drop( LABEL )

    equal_count = 0
    if len( h.vals ) != len( sample ):
        return False

    for i in range( len( sample ) ):

        if h[ i ] == SPECIFIC:
            return False

        if sample[ i ] == h[ i ] or h[ i ] == GENERAL:
            equal_count += 1

    if label == "negative":
        if len( sample ) == equal_count:
            return False
        return True

    if label == "positive":
        if len( sample ) == equal_count:
            return True
        return False


def consistent( h: Hypothesis, d: Dataset ) -> bool:

    for index, sample in d.iterrows():
        if not sample_consistent( h, sample ):
            return False

    return True


example = pd.read_excel( "example.xlsx" )
h = Hypothesis( ["?", "?", "circle"] )


attr_pool = [ ["small", "large"], [ "red", "blue" ], [ "circle", "triangle" ] ]


## Hypothesis space
def all_hypothesis( attr_pool: list[ list[ Attribute_value ] ] ) -> list[ Hypothesis ]:
    
    possible = [ GENERAL for _ in range( len( attr_pool ) ) ]
    res = []
    all_hypothesis_rec( possible, 0, attr_pool, res )
    possible = [ SPECIFIC for _ in range( len( attr_pool ) ) ]
    res.append( Hypothesis( possible[:] ) )

    return res

def all_hypothesis_rec( current: list[ Attribute_value ],
                        i: int,
                        attr_pool: list[ list[ Attribute_value ] ],
                        res: list[ Hypothesis ] ) -> None:

    if i == len( attr_pool ):
        res.append( Hypothesis( current[:] ) )
        return

    for val in [GENERAL] + attr_pool[ i ]:
        current[ i ] = val
        all_hypothesis_rec( current, i + 1, attr_pool, res )


H = all_hypothesis( attr_pool )

def consistent_hypothesis( H: list[ Hypothesis ], D: Dataset ) -> Hypothesis:
    for h in H:
        if consistent( h, D ):
            return h


## Complete subset of hypothesis space that is consistent with dataset --> version space

def naive_version_space( H: list[ Hypothesis ], D: Dataset ) -> list[ Hypothesis ]:
    return [ h for h in H if consistent( h, D ) ]
