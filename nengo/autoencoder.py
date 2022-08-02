"""
An autoencoder experiment to see if we can do localised learning.

Resolved:
1. Dimensions for transform is ok

Current problems:
1. Credit assignment of reconstruction loss.
2. Unable to debug on UI since it continuously crashes.
"""

import nengo
import numpy as np
import tensorflow as tf
from nengo_extras.gui import image_display_function
from nengo_extras.vision import Gabor, Mask

model = nengo.Network()

rng = np.random.RandomState(9)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
img_dim = X_train[0].reshape(-1).shape[0]
hidden_dim = 392

def preprocess(img):
    X = img.reshape(-1)
    X = X - np.mean(X)
    return X / np.linalg.norm(X)

with model:
    # Gabor encoders are used as transforms from input dimension to hidden dimension
    encoders = Mask((28,28)).populate(
        Gabor().generate(hidden_dim, (11,11), rng=rng),
        rng=rng, 
        flatten=True
    )

    # MNIST input feed
    input = nengo.Node(lambda t: preprocess(X_train[int(t)]))

    # Neural representation of MNIST image
    in_ensemble = nengo.Ensemble(
        n_neurons=6000,
        dimensions=img_dim,
    )

    nengo.Connection(input, in_ensemble)

    # Hidden layer
    hidden = nengo.Ensemble(
        n_neurons=5000,
        dimensions=hidden_dim
    )

    # Feedforward encoding connection
    # We want to learn f(x) using PES learning rule.
    conn1 = nengo.Connection(
        in_ensemble, hidden, 
        transform=encoders,
        learning_rule_type=nengo.PES()
    )

    out_ensemble = nengo.networks.EnsembleArray(
        n_neurons=30,
        n_ensembles=img_dim,
        ens_dimensions=1
    )
    
    # Feedforward decoding connection
    # These are solved using NEF by evaluating on the MNIST points
    eval_points = np.dot(X_train.reshape(len(X_train), -1), encoders.T)
    conn2 = nengo.Connection(
        hidden, out_ensemble.input,
        eval_points=eval_points,
        function=X_train.reshape(len(X_train), -1)
    )

    # Error signal calculation
    # and error signal connection
    recon_error = nengo.Ensemble(
        n_neurons=784,
        dimensions=hidden_dim
    )
    nengo.Connection(out_ensemble.output, recon_error, transform=-1*encoders)
    nengo.Connection(input, recon_error, transform=encoders)
    nengo.Connection(
        recon_error, conn1.learning_rule
    )

    # Input image display (for nengo_gui)
    image_shape = (1, 28, 28)
    display_func = image_display_function(image_shape, offset=1, scale=128)
    input_disp = nengo.Node(display_func, size_in=input.size_out)
    nengo.Connection(input, input_disp, synapse=None)

    output = nengo.Node(display_func, size_in=img_dim)
    nengo.Connection(out_ensemble.output, output, synapse=0.1)
    
    
