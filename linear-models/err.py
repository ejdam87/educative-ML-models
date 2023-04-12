from vector import Vector

## mean squared error
def mse( actual: list[ float ], expected: list[ float ] ) -> float:

    assert len( actual ) == len( expected )

    n = len( actual )
    res = 0
    for a, b in zip( actual, expected ):
        res += ( b - a ) ** 2

    return res / n

## partial derivation ( by k-th elementh of weight vector )
## of mse
def dmse( k: int,
          inputs: list[ Vector ],
          actual: list[ float ],
          expected: list[ float ] ) -> float:

    assert len( actual ) == len( expected ) == len( inputs )

    n = len( actual )
    res = 0
    for a, b, inpt in zip( actual, expected, inputs ):
        res += ( b - a ) * inpt[ k ]

    return ( -2 / n ) * res
