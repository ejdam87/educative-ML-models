import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Callable

def identity(x: float) -> float:
    return x

def didentity(x: float) -> float:
    return 1

def sigmoid(x: float) -> float:
    return 1 / (1 + np.e**(-x))

def dsigmoid(x: float) -> float:    # for GD purposes
    return sigmoid(x) * (1 - sigmoid(x))

def mse(preds: np.array, targets: np.array) -> float:
    return (1 / 2) * np.sum(np.square(preds - targets))

def dmse(preds: np.array, targets: np.array) -> float:  # derivative w.r.t preds
    return preds - targets

vsigmoid = np.vectorize(sigmoid)
vdsigmoid = np.vectorize(dsigmoid)
videntity = np.vectorize(identity)
vdidentity = np.vectorize(didentity)


class DenseLayer:
    def __init__(self, dim_in: int, dim_out, activation: Callable[[float], float]) -> None:
        self.weights = matrix = np.random.rand(dim_out, dim_in)
        self.biases = np.random.rand(dim_out, 1)
        self.activation = activation

    def forward(self, x: NDArray) -> NDArray:
        prod = self.weights @ x + self.biases
        return self.activation(prod), prod # last multiplication column simulates bias addition

    def eval(self, x: NDArray) -> int:
        y, _ = self.forward(x)
        return np.where(y >= 0.5, 1, 0)


class MLP:
    def __init__(self, layer_sizes: np.array, activations: np.array, dactivations: np.array) -> None:
        assert len(layer_sizes) - len(activations) == 1

        self.dactivations = dactivations
        # --- sequence of dense layer with respective output activation
        self.layers = []
        for i, (dim_in, dim_out) in enumerate(zip(layer_sizes, layer_sizes[1:])):
            self.layers.append( DenseLayer( dim_in, dim_out, activations[i]) )
        # ---

    def forward(self, x: NDArray, preserve_activations: bool=False, preserve_dots: bool=False) -> tuple[NDArray, NDArray]:
        A = []
        Z = []
        for i, layer in enumerate(self.layers):
            if preserve_activations:
                A.append(x)

            x, z = layer.forward(x)
            if preserve_dots:
                Z.append(z)

        if preserve_activations:
            A.append(x)

        return x, A, Z

    def eval(self, x: NDArray) -> NDArray:
        y, _, _ = self.forward(x)
        return np.where(y >= 0.5, 1, 0)


def grad(model: MLP,
         X: NDArray,
         y: np.array,
         lr=0.01,
         epochs=100) -> list[float]:   # we modify mdoel weights in-place

    loss = []
    for _ in range(epochs):

        # --- forward pass
        Y_pred, A, Z = model.forward(X, True, True) # evaluating all examples at once
        loss.append( mse(Y_pred, y) )

        # --- backpropagation
        prev_dl = None
        # passing layers backwards
        for i in range( len(model.layers) - 1, -1, -1 ):
            if i == len(model.layers) - 1:
                dl = dmse(Y_pred, y) * model.dactivations[i](Z[i])
                prev_dl = dl
            else:
                dl = ((model.layers[i + 1].weights.T) @ prev_dl) * model.dactivations[i](Z[i])
                prev_dl = dl

            wgradient = dl @ A[i].T
            bgradient = np.sum(dl, axis=1)
            bgradient = bgradient.reshape(-1, 1)
            model.layers[i].weights -= lr * wgradient
            model.layers[i].biases -= lr * bgradient

    return loss
