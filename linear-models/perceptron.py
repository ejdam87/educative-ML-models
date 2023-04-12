import random
from vector import Vector
from typing import Callable

class Perceptron:

    def __init__( self,
                  dim: int,
                  activation: Callable[ [ float ], float ]=None ) -> None:

        self.dim = dim
        self.val = random.random( )
        self.weights = Vector( [ random.random( ) for _ in range( dim + 1 ) ] )
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

        return v.extended( -1 ) * self.weights

    def compute( self, v: Vector ) -> None:
        assert self.activation is not None, "Missing activation function"

        sm = self.weighted_sum( v )
        self.val = self.activation( sm )
