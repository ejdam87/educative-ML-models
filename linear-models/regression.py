from vector import Vector
import err
import random

## ( input, expected_result )
Sample = tuple[ Vector, float ]


class LinearRegression:
    def __init__( self, dim: int ) -> None:
        self.weights = Vector( [ random.random( ) for _ in range( dim + 1 ) ] )    ## bias is weights[ 0 ]
        self.dim = dim
        self.learning_rate = 0.001

    ## Main method --> try to approximate expected value ( based on current weights )
    def approximate( self, v: Vector ) -> float:
        return self.weights * v.extended( 1 )

    def set_weights( self, weights: Vector ) -> None:
        self.weights = weights

    def get_weights( self ) -> Vector:
        return self.weights

    ## Error gradient calculation
    def gradient( self, samples: list[ Sample ] ) -> Vector:

        inputs = [ inpt.extended( 1 ) for inpt, expected in samples ]
        res = Vector( [ 0 for _ in range( self.dim + 1 ) ] )

        actual, expected = self._compute_approx( samples )
        for k in range( self.dim + 1 ):
            res[ k ] = err.dmse( k, inputs, actual, expected )

        return res

    def _compute_approx( self, samples: list[ Sample ] ) -> tuple[ list[ float ], list[ float ] ]:
        actual = []
        expected = []

        for v, exp in samples:
            actual.append( self.approximate( v ) )
            expected.append( exp )

        return actual, expected

    def regression_error( self, samples: list[ Sample ] ) -> float:
        actual, expected = self._compute_approx( samples )
        return err.mse( actual, expected )

    def learn( self, samples: list[ Sample ] ) -> None:

        EPOCHS = 10000
        for i in range( EPOCHS ):
            self.weights -= self.learning_rate * self.gradient( samples )

