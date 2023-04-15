import frame as fr
from collections import defaultdict
from typing import TypeVar


## No smoothing applied
## !!! We assume discrete features !!!

Label = TypeVar( "Label" )

"""
P( y | x ) = ( P( x | y ) * P( y ) ) / P( x )

P( x | y ) = (!!NAIVITY!!) = prod( P( x_i | y ) for all features )

P( x_i | y ) = P( x_i and y ) / P( y )
"""

class NaiveBayesClassifier:
    """
    Almost the same as bayes classifier
    Difference is that we assume conditional independence of features
    """
    def __init__( self ) -> None:
        self.probs = {}
        self.labels = set()
        self.y_probs = {}
        self.columns = []

    def get_prob( self, y: Label, feature: str, val: float ) -> float:
        
        for k, v in self.probs.items():
            if k == ( y, feature, val ):
                return v

        ## unseen
        return 0

    def learn( self, data: fr.Frame, labels: fr.Frame ) -> None:

        ## Calculate P( x_i | y )
        counts = defaultdict( int )
        y_counts = defaultdict( int )
        for sample, label in zip( data.rows(), labels.rows() ):
            label = label[0]

            y_counts[ label ] += 1

            for feature, name in zip( sample, data.columns ):
                counts[ (label, name, feature) ] += 1

        probs = {}
        for ( label, name, feature ), count in counts.items():
            probs[ (label, name, feature) ] = count / y_counts[ label ]

        self.probs = probs
        self.labels = set( y_counts )
        self.columns = data.columns

        n = len( data.rows() )
        self.y_probs = { y: count / n for y, count in y_counts.items() }
            

    def predict( self, sample: fr.Sample ) -> Label | None:

        ## calculate naively P( x | y ) for every seen value of y
        cond_y_probs = defaultdict(lambda: 1)
        for label in self.labels:
            for feature_val, name in zip( sample, self.columns ):

                ## Here we assume features are conditionally independent
                cond_y_probs[ label ] *= self.get_prob( label, name, feature_val )

        ## use bayes rule and find biggest prob (we do not need P(x) bcs it's the same for all probs)
        max_prob = -1
        res = None

        for label in self.labels:
            prob = cond_y_probs[ label ] * self.y_probs[ label ]
            if prob > max_prob:
                max_prob = prob
                res = label

        return res


data = [
        ["X1", "X2", "X3"],
        [ 1, 0, 1 ],
        [ 0, 1, 1 ],
        [ 1, 0, 1 ],
        [ 0, 0, 1 ],
        [ 0, 0, 0 ],
        [ 1, 1, 1 ],
       ]

labels = [
            ["Y"],
            [1],
            [1],
            [0],
            [0],
            [1],
            [0]
         ]

nb = NaiveBayesClassifier()
nb.learn( fr.Frame(data), fr.Frame(labels) )
print( nb.predict( [ 0, 0, 0 ] ) )