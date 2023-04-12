from aliases import *
import matplotlib.pyplot as plt

def show_2D_dataset( dataset: Dataset ) -> None:
    assert len( dataset.columns ) == 2 + 1 ## Also label
    x, y, label = dataset.columns

    plt.scatter( data=dataset, x=x, y=y )
    plt.show()

def show_predicting_in_data( dataset: Dataset, predicting: Sample ) -> None:
    x, y, label = dataset.columns

    plt.scatter( data=dataset, x=x, y=y, c=label, label=label )
    plt.scatter( x=predicting[ 0 ], y=predicting[ 1 ], c="green" )

    plt.show()  
