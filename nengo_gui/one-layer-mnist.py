import nengo
import numpy as np
import tensorflow as tf
from nengo_extras.gui import image_display_function

# Load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Experiment hyperparameters
n_neurons = 6000
input_size = X_train[0].reshape(-1).shape[0]
hidden_dim = input_size // 2

def preprocess(X):
    # Preprocess data to nengo node input to unit vectors
    # Subtract the mean to make better use of full representational power of ensembles
    X = X.reshape(-1)
    X = X - np.mean(X)
    return X / np.linalg.norm(X)

# Building the network
model = nengo.Network(label="mnist")

with model:
    vision_input = nengo.Node(lambda t: preprocess(X_train[int(t) // 2]), label="Visual Input")
    input_ensemble = nengo.Ensemble(
        n_neurons=n_neurons,
        dimensions=input_size,
        radius=1
    )
    nengo.Connection(
        vision_input, input_ensemble
    )

    # Input image display (for nengo_gui)
    image_shape = (1, 28, 28)
    display_func = image_display_function(image_shape, offset=1, scale=128)
    display_node = nengo.Node(display_func, size_in=vision_input.size_out)
    nengo.Connection(vision_input, display_node, synapse=None)

    output = nengo.Node(display_func, size_in=784)
    nengo.Connection(input_ensemble, output, synapse=0.1)