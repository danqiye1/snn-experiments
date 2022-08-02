"""
An Ensemble Array Representation of MNIST.
"""

import nengo
import numpy as np
import tensorflow as tf
from nengo_extras.gui import image_display_function

# Load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
img_dim = X_train[0].reshape(-1).shape[0]

model = nengo.Network(label="Input Ensemble Array")

def preprocess(img):
    X = img.reshape(-1)
    X = X - np.mean(X)
    return X / np.linalg.norm(X)

with model:
    vision_input = nengo.Node(lambda t: preprocess(X_train[int(t) // 2]), label="Visual Input")

    input_ensemble = nengo.networks.EnsembleArray(
        n_neurons=30,
        n_ensembles=img_dim,
        ens_dimensions=1
    )

    nengo.Connection(
        vision_input, input_ensemble.input
    )

    # Input image display (for nengo_gui)
    image_shape = (1, 28, 28)
    display_func = image_display_function(image_shape, offset=1, scale=128)
    display_node = nengo.Node(display_func, size_in=vision_input.size_out)
    nengo.Connection(vision_input, display_node, synapse=None)

    output = nengo.Node(display_func, size_in=784)
    nengo.Connection(input_ensemble.output, output, synapse=0.1)