"""
MNIST PES Learning

We attempt to use PES learning rule to learn the MNIST classification.
"""
import nengo
import numpy as np
from utils import preprocess
import tensorflow as tf
from nengo_extras.gui import image_display_function
from nengo_extras.data import one_hot_from_labels
from nengo_extras.vision import Gabor, Mask

# Load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
T_train = one_hot_from_labels(y_train)
input_size = X_train[0].reshape(-1).shape[0]

model = nengo.Network(label='mnist')

rng = np.random.RandomState(9)

with model:
    vision_input = nengo.Node(lambda t: preprocess(X_train[int(t)]), label="Visual Input")

    encoders = Mask((28,28)).populate(
        Gabor().generate(10, (11,11), rng=rng),
        rng=rng, 
        flatten=True
    )
    # Ensemble to encode MNIST images
    input_ensemble = nengo.Ensemble(
        n_neurons=6000,
        dimensions=input_size,
        radius=1
    )
    nengo.Connection(
        vision_input, input_ensemble
    )

    # Ensemble to encode MNIST labels
    output_ensemble = nengo.Ensemble(
        n_neurons=300,
        dimensions=10,
        radius=5
    )

    conn = nengo.Connection(
        input_ensemble, output_ensemble,
        learning_rule_type=nengo.PES(),
        transform=encoders
    )

    error = nengo.Ensemble(
        n_neurons=300,
        dimensions=10,
        radius=5
    )
    label=nengo.Node(lambda t: T_train[int(t)], label="Digit Labels")
    nengo.Connection(output_ensemble, error)
    nengo.Connection(label, error, transform=-1)
    nengo.Connection(error, conn.learning_rule)

    # Input image display (for nengo_gui)
    image_shape = (1, 28, 28)
    display_func = image_display_function(image_shape, offset=1, scale=128)
    display_node = nengo.Node(display_func, size_in=vision_input.size_out)
    nengo.Connection(vision_input, display_node, synapse=None)

    inp_ens_disp = nengo.Node(display_func, size_in=784)
    nengo.Connection(input_ensemble, inp_ens_disp, synapse=0.1)

