import random
from vector import Vector
from typing import Callable, TypeVar

Label = TypeVar( "T" )


class Perceptron:

    def __init__( self,
                  dim: int,
                  activation: Callable[ [ float ], float ]=None ) -> None:

        self.learning_rate = 1

        self.dim = dim
        self.val = random.random( )
        self.weights = Vector( [ 0 for _ in range( dim + 1 ) ] )
        self.activation = activation

    def set_weights( self, v: Vector ) -> None:
        self.weights = v

    def get_weights( self ) -> Vector:
        return self.weights

    def get_dim( self ) -> int:
        return self.dim

    def get_value( self ) -> float:
        return self.val

    def set_value( self, val: float ) -> None:
        self.val = val

    def set_activation( self, ac: Callable[ [ float ], float ] ) -> None:
        self.activation = activation

    def weighted_sum( self, v: Vector ) -> float:
        assert len( v ) == len( self.weights ) - 1, "Invalid vector size"

        return v.extended( 1 ) * self.weights

    def compute( self, v: Vector ) -> None:
        assert self.activation is not None, "Missing activation function"

        sm = self.weighted_sum( v )
        self.val = self.activation( sm )

    def learn( self, data: list[ Vector ], labels: list[ Label ] ) -> None:
        assert len( data ) == len( labels )

        ## We want consistent separator
        converged = False

        ## It is PROVEN that perceptron learning algorithm will always converge
        while not converged:

            converged = True
            for sample, label in zip( data, labels ):
                predicted = self.predict( sample )
                
                if ( predicted - label ) != 0:
                    converged = False

                self.weights = self.weights - self.learning_rate * ( predicted - label ) * sample.extended( 1 )


    def predict( self, v: Vector ) -> Label:
        return 1 if self.weighted_sum( v ) >= 0 else 0
