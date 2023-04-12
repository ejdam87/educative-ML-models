from aliases import *
from collections import defaultdict
import show
from distance import euclidean


LABEL = "label"  ## State what attribute is label

class KnnClassifier:
    def __init__( self, k: int ) -> None:
        """
        k - states how many neighbours to take into the account
        """
        self.k = k
        self.samples = []

    def learn( self, data: Dataset ) -> None:
        """
        Method to learn the model
        """
        assert len( data ) >= self.k  ## If we have fewer samples than neighbours to check --> impossible
        self.samples = data

    def closest_k( self, sample: Sample ) -> list[ tuple[Sample, float] ]:
        """
        Method to find the closest k samples to given sample ( k was provided during initialization )
        """
        distances = []
        for _, memorized in self.samples.iterrows():
            distance = euclidean( memorized.drop( LABEL ), sample )
            distances.append( ( memorized, distance ) )

        distances = sorted( distances, key=lambda x: x[1] )
        return distances[ :self.k ]

    def mode_label( self, samples: list[ Sample ] ) -> Attribute_value:
        """
        Method to get the most frequent label in samples
        """
        ctr = defaultdict(int)
        for sample in samples:
            ctr[ sample[ LABEL ] ] += 1

        return max( ctr, key=lambda x: x[1] )


    def predict( self, sample: Sample ) -> Attribute_value:
        """
        Method to predict sample's label
        """
        closest = self.closest_k( sample )
        records = [ record for record, _ in closest ]
        return self.mode_label( records )


class KnnRegressor( KnnClassifier ):
    def __init__( self, k: int ) -> None:
        super().__init__( k )

    def predict( self, sample: Sample ) -> float:
        closest = self.closest_k( sample )
        records = [ record for record, _ in closest ]
        return self.mean_label( records )

    def mean_label( self, samples: list[ Sample ] ) -> float:
        labels = [ sample[ LABEL ] for sample in samples ]
        return sum( labels ) / len( labels )
