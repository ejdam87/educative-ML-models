from vector import Vector
from regression import LinearRegression
import datagen
import matplotlib.pyplot as plt

import random


## ( input, expected_result )
Sample = tuple[ Vector, float ]

def vectorify( *args: int ) -> Vector:
    return Vector( list( args ) )


def show_2d_regression( data: list[ Sample ],
                        model: LinearRegression ) -> None:
    
    assert model.dim == 1

    xs = [ x[ 0 ] for x, _ in data ]
    ys = [ y for _, y in data ]

    plt.scatter( xs, ys, color="b" )

    ys = [ model.approximate( vectorify( x ) ) for x in xs ]
    plt.plot( xs, ys, color="g" )
    plt.show( )


def show_3d_regression( data: list[ Sample ],
                        model: LinearRegression ) -> None:
    
    assert model.dim == 2

    xs = [ x[ 0 ] for x, _ in data ]
    ys = [ x[ 1 ] for x, _ in data ]
    zs = [ z for _, z in data ]

    fig = plt.figure()
    ax = fig.add_subplot( projection="3d" )

    ax.scatter( xs, ys, zs, color="b" )

    zs = [ model.approximate( vectorify( x, y ) ) for x, y in zip( xs, ys ) ]
    ax.plot_trisurf( xs, ys, zs, color="g" )
    plt.show( )

model = LinearRegression( 2 )
dataset = datagen.create_dataset( 15, 2 )
model.learn( dataset )
show_3d_regression( dataset, model )
