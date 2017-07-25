
# coding: utf-8

# # Nengo Example: A Simple Harmonic Oscillator
# This demo implements a simple harmonic oscillator in a 2D neural population. The oscillator is more visually interesting on its own than the integrator, but the principle at work is the same. Here, instead of having the recurrent input just integrate (i.e. feeding the full input value back to the population), we have two dimensions which interact. In Nengo there is a ‘Linear System’ template which can also be used to quickly construct a harmonic oscillator (or any other linear system).

# In[ ]:

import matplotlib.pyplot as plt

import nengo


# ## Step 1: Create the Model
# The model consists of a single neural ensemble that we will call `Neurons`.

# In[ ]:

# Create the model object
model = nengo.Network(label='Oscillator')
with model:
    # Create the ensemble for the oscillator
    neurons = nengo.Ensemble(200, dimensions=2)


# ## Step 2: Provide Input to the Model
# A brief input signal is provided to trigger the oscillatory behavior of the neural representation.

# In[ ]:

from nengo.utils.functions import piecewise
with model:
    # Create an input signal
    input = nengo.Node(piecewise({0: [1, 0], 0.1: [0, 0]}))

    # Connect the input signal to the neural ensemble
    nengo.Connection(input, neurons)

    # Create the feedback connection
    nengo.Connection(neurons, neurons, transform=[[1, 1], [-1, 1]], synapse=0.1)


# ## Step 3: Add Probes
# These probes will collect data from the input signal and the neural ensemble.

# In[ ]:

with model:
    input_probe = nengo.Probe(input, 'output')
    neuron_probe = nengo.Probe(neurons, 'decoded_output', synapse=0.1)


# ## Step 4: Run the Model

# In[ ]:

# Create the simulator
with nengo.Simulator(model) as sim:
    # Run it for 5 seconds
    sim.run(5)


# ## Step 5: Plot the Results

# In[ ]:

plt.figure()
plt.plot(sim.trange(), sim.data[neuron_probe])
plt.xlabel('Time (s)', fontsize='large')
plt.legend(['$x_0$', '$x_1$']);


# In[ ]:

data = sim.data[neuron_probe]
plt.figure()
plt.plot(data[:, 0], data[:, 1], label='Decoded Output')
plt.xlabel('$x_0$', fontsize=20)
plt.ylabel('$x_1$', fontsize=20)
plt.legend();


plt.show()
