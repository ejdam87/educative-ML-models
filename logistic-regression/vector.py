from typing import TypeVar, Iterator, Callable

T = TypeVar( "T" )

class Vector:
    def __init__( self, data: list[ T ] ) -> None:
        self.data = data

    def __repr__( self ) -> str:
        str_data = ", ".join( str( e ) for e in self.data )
        return f"< {str_data} >"

    def __iter__( self ) -> Iterator[ T ]:
        for e in self.data:
            yield e

    def __len__( self ) -> int:
        return len( self.data )

    def operation( self, other: "Vector", op: Callable[ [ T, T ], T ] ) -> "Vector":
        res = []
        for a, b in zip( self, other ):
            res.append( op( a, b ) )

        return Vector( res )

    def _dot( self, other: "Vector" ) -> T:
        res = 0
        for a, b in zip( self, other ):
            res += a * b
        return res

    def __add__( self, other: "Vector" ) -> "Vector":
        return self.operation( other, lambda a, b: a + b )

    def __sub__( self, other: "Vector" ) -> "Vector":
        return self.operation( other, lambda a, b: a - b )

    def __mul__( self, other: "Vector" | T ) -> "Vector" | T:

        if isinstance( other, Vector ):
            return self._dot( other )

        return Vector( [ other * e for e in self.data ] )

    def __rmul__( self, other: "Vector" | T ) -> "Vector" | T:
        return self.__mul__( other )

    def extended( self, val: T ) -> "Vector":
        return Vector( [ val ] + self.data )

    def __getitem__( self, i: int ) -> T:
        return self.data[ i ]

    def __setitem__( self, i: int, item: T) -> None:
        self.data[ i ] = item
