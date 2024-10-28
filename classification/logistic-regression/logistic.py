from math import e
from vector import Vector
from typing import TypeVar

Label = TypeVar( "T" )

class LogisticRegression:

    def __init__( self, n: int ) -> None:
        self.n = n
        self.weights = Vector( [ 0 for i in range( n + 1 ) ] )
        self.learning_rate = 0.07
        self.treshold = 1 / 2

    def error_gradient( self,
                        samples: list[ Vector ],
                        labels: list[ Label ] ) -> Vector:
        
        res = Vector( [ 0 for _ in range( self.n + 1 ) ] )
        for sample, label in zip( samples, labels ):
            current = ( self.get_prob( sample ) - label ) * sample.extended( 1 )
            res += current

        return res


    def get_prob( self, sample: Vector ) -> float:
        dot = self.weights * sample.extended( 1 )
        return 1 / ( 1 + e**( -dot ) )


    def predict( self, sample: Vector ) -> Label:
        return 0 if self.get_prob( sample ) < self.treshold else 1

    def learn( self, samples: list[ Vector ], labels: list[ Label ] ) -> None:
        epochs = 10000
        for _ in range( epochs ):
            self.weights = self.weights - self.learning_rate * self.error_gradient( samples, labels )


samples = [
            Vector( [0] ),
            Vector( [0.5] ),
            Vector( [1] ),
            Vector( [0.4] ),

            Vector( [11] ),
            Vector( [12] ),
            Vector( [13] )
          ]

labels = [ 0, 0, 0, 0, 1, 1, 1 ]

lr = LogisticRegression( 1 )
lr.learn( samples, labels )
