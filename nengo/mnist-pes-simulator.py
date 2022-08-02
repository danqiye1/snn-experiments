"""
MNIST PES Learning

In this experiment, we attempt to use PES learning rule to learn the MNIST classification.
The learning rule should be turned off after t_off seconds to visualize if the weights have been learned properly.
"""
import nengo
import numpy as np
from utils import preprocess
import tensorflow as tf
from nengo_extras.data import one_hot_from_labels
from nengo_extras.vision import Gabor, Mask
from matplotlib import pyplot as plt

# Load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
T_train = one_hot_from_labels(y_train)
input_size = X_train[0].reshape(-1).shape[0]

model = nengo.Network(label='mnist')

# Hyperparameters
rng = np.random.RandomState(9)
t_off = 100

with model:
    # MNIST input
    vision_input = nengo.Node(lambda t: preprocess(X_train[int(t)]), label="Visual Input")

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

    # Gabor filters are used to transform
    # vector of dim=784 to dim=10
    encoders = Mask((28,28)).populate(
        Gabor().generate(10, (11,11), rng=rng),
        rng=rng, 
        flatten=True
    )
    conn = nengo.Connection(
        input_ensemble, output_ensemble,
        learning_rule_type=nengo.PES(),
        transform=encoders
    )

    # Error signal
    error = nengo.Ensemble(
        n_neurons=300,
        dimensions=10,
        radius=5
    )
    label=nengo.Node(lambda t: T_train[int(t)], label="Digit Labels")
    nengo.Connection(output_ensemble, error)
    nengo.Connection(label, error, transform=-1)
    nengo.Connection(error, conn.learning_rule)

    # Error inhibition
    error_inhibit = nengo.Node(lambda t: t >= t_off)
    nengo.Connection(error_inhibit, error.neurons, transform=-100*np.ones((error.n_neurons, 1)))

    # Probe the predicted labels
    prediction_probe = nengo.Probe(output_ensemble, synapse=0.1)

with nengo.Simulator(model) as sim:
    sim.run(150)

for i in range(10):
    plt.plot(sim.trange(), sim.data[prediction_probe][:,i], label=f"{i}")
plt.savefig("tmp.png")