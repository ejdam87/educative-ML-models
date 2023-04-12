import frame as fr
from collections import defaultdict
from typing import TypeVar

## !!! We assume discrete features !!!

Label = TypeVar( "Label" )

class BayesClassifier:

    """
    P( class | given features )
    """
    def __init__( self ) -> None:
        self.probs = {}

    def get_prob( self, label: Label, sample: fr.Sample ) -> float:
        return self.probs[ ( label, sample ) ]

    def learn( self, data: fr.Frame, labels: fr.Frame ) -> None:

        counts = defaultdict(int)
        row_count = defaultdict(int)

        labels_rows = labels.rows()
        for i, row in enumerate( data.rows() ):
            label = labels_rows[ i ][0]
            counts[ ( label, tuple(row) ) ] += 1
            row_count[ tuple(row) ] += 1

        for label, row in counts:
            self.probs[ (label, row) ] = counts[ (label, row) ] / row_count[ row ]

    def predict( self, sample: fr.Sample ) -> Label:

        res = None
        curr_max = -1
        for (label, present_sample), prob in self.probs.items():
            if present_sample == tuple(sample) and curr_max < prob:
                curr_max = prob
                res = label

        return res
