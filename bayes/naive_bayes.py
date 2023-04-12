import frame as fr
from collections import defaultdict
from typing import TypeVar

## !!! We assume discrete features !!!

Label = TypeVar( "Label" )


class NaiveBayesClassifier:
    """
    Almost the same as bayes classifier
    Difference is that we assume conditional independence of features
    """
    def __init__( self ) -> None:
        pass

    def learn( self, data: fr.Frame, labels: fr.Frame ) -> None:
        pass

    def predict( self, sample: fr.Sample ) -> Label:
        pass
