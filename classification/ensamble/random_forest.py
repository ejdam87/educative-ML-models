import pandas as pd
import decision
import random

from collections import defaultdict

Label = str

class RandomForestClassifier:
    """
    Represents a simple educative version of random forest classifier
    """
    def __init__( self, n: int ) -> None:
        self.tree_count = n

        ## tree -> list of indices of used attributes
        self.trees: dict[ decision.DecisionTree, list[ int ] ] = {}

    def learn( self, data: pd.DataFrame, labels: pd.DataFrame ) -> None:
        ## For every tree, select random k attributes and learn the tree on them
        
        attr_count = len( data.columns )

        for _ in range( self.tree_count ):
            k = random.randint( 1, attr_count )
            attributes = random.sample( list(data.columns), k )
            subframe = data[ attributes ].join( labels )
            tree = decision.decision_tree( subframe, labels.name )
            self.trees[ tree ] = [ data.columns.get_loc( attr ) for attr in attributes ]

    def predict( self, sample: pd.Series ) -> Label:
        ## performs prediciton based on majority-voted (e.g. most common outcome)
        counts = defaultdict( int )
        for tree, attribute_i in self.trees.items():

            selected_sample = [ sample[ i ] for i in attribute_i ]
            counts[ tree.predict( selected_sample ) ] += 1

        return max( counts.items(), key=lambda x: x[1] )[0]


data = pd.read_excel( "tenis.xlsx" )
rf = RandomForestClassifier( n=10 )
rf.learn( data.drop( "Tenis", axis=1 ), data[ "Tenis" ] )

dt = decision.decision_tree( data, "Tenis" )

sample = [ "high", "sunny", "no", "lower" ]
print( dt.predict( sample ) )
print( rf.predict( sample ) )
