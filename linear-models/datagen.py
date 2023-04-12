from vector import Vector
from random import randint

## ( input, expected_result )
Sample = tuple[ Vector, float ]

def create_x( dim: int, lower: int, upper: int ) -> Vector:
    return Vector( [ randint( lower, upper ) for _ in range( dim ) ] )

## Sample "almost" linear function
def linear( xs: Vector, variance: int ) -> int:

    res = 0
    for x in xs:
        res += x

    return res + randint( -variance, variance )


def create_dataset( size: int,
                    dim: int,
                    lower_x: int=-10,
                    upper_x: int=10,
                    lower_y: int=-10,
                    upper_y: int=10,
                    variance: int=5 ) -> list[ Sample ]:
    
    res = []
    for _ in range( size ):
        
        x = create_x( dim, lower_x, upper_x )
        y = linear( x, variance )
        res.append( ( x, y ) )

    return res
