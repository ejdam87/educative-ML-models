from typing import TypeVar

T = TypeVar( "T" )
Sample = list[ T ]
Grid = list[ Sample ]

class Frame:
    def __init__( self, data: Grid=None ) -> None:
        self.columns = data[ 0 ]
        self.data = data[ 1: ]

    def rows( self ) -> Grid:
        return self.data

    def __repr__( self ) -> str:

        res = " ".join( self.columns ) + "\n"
        res += "---" + "\n"

        for row in self.data:
            res += " ".join( [ str( x ) for x in row ] ) + "\n"

        return res

"""
data = [
        ["A", "B", "C"],
        [ 1, 2, 3 ],
        [ 4, 5, 6 ] 
       ]

f = Frame( data )
print( f )
"""
