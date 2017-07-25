
# coding: utf-8

# # Nengo example: Simple question answering
# 
# This demo implements a simple form of question answering. Two features (color and shape) will be bound by circular convolution. A cue will be used to determine either one of the features by deconvolution.
# 
# When you run the network, it will start by binding `RED` and `CIRCLE` for 0.5 seconds and then binding `BLUE` and `SQUARE` for 0.5 seconds. In parallel the network is asked with the cue. For example, when the cue is `CIRCLE` the network will respond with `RED`.

# In[ ]:

import matplotlib.pyplot as plt

import nengo
from nengo import spa


# ## Step 1: Create the model

# In[ ]:

# Number of dimensions for the Semantic Pointers
dimensions = 32

model = spa.SPA(label="Simple question answering")

with model:
    # initialise the state populations
    model.color_in = spa.State(dimensions=dimensions)
    model.shape_in = spa.State(dimensions=dimensions)
    model.conv = spa.State(dimensions=dimensions)
    model.cue = spa.State(dimensions=dimensions)
    model.out = spa.State(dimensions=dimensions)

    # Connect the state populations
    cortical_actions = spa.Actions(
        'conv = color_in * shape_in',
        'out = conv * ~cue'
    )
    model.cortical = spa.Cortical(cortical_actions)


# ## Step 2: Provide the input
# 
# The color input will switch every 0.5 seconds between `RED` and `BLUE`. In the same way the shape input switches between `CIRCLE` and `SQUARE`. Thus, the network will bind alternatingly `RED * CIRCLE` and `BLUE * SQUARE` for 0.5 seconds each.
# 
# The cue for deconvolving bound semantic pointers cycles through `CIRCLE`, `RED`, `SQUARE`, and `BLUE` within one second. 

# In[ ]:

def color_input(t):
    if (t // 0.5) % 2 == 0:
        return 'RED'
    else:
        return 'BLUE'


def shape_input(t):
    if (t // 0.5) % 2 == 0:
        return 'CIRCLE'
    else:
        return 'SQUARE'


def cue_input(t):
    sequence = ['0', 'CIRCLE', 'RED', '0', 'SQUARE', 'BLUE']
    idx = int((t // (1. / len(sequence))) % len(sequence))
    return sequence[idx]


with model:
    model.inp = spa.Input(
        color_in=color_input, shape_in=shape_input, cue=cue_input)


# ## Step 3: Probe the output

# In[ ]:

with model:
    model.config[nengo.Probe].synapse = nengo.Lowpass(0.03)
    color_in = nengo.Probe(model.color_in.output)
    shape_in = nengo.Probe(model.shape_in.output)
    cue = nengo.Probe(model.cue.output)
    conv = nengo.Probe(model.conv.output)
    out = nengo.Probe(model.out.output)


# ## Step 4: Run the model

# In[ ]:

with nengo.Simulator(model) as sim:
    sim.run(3.)


# ## Step 5: Plot the results

# In[ ]:

plt.figure(figsize=(10, 10))
vocab = model.get_default_vocab(dimensions)

plt.subplot(5, 1, 1)
plt.plot(sim.trange(), model.similarity(sim.data, color_in))
plt.legend(model.get_output_vocab('color_in').keys, fontsize='x-small')
plt.ylabel("color")

plt.subplot(5, 1, 2)
plt.plot(sim.trange(), model.similarity(sim.data, shape_in))
plt.legend(model.get_output_vocab('shape_in').keys, fontsize='x-small')
plt.ylabel("shape")

plt.subplot(5, 1, 3)
plt.plot(sim.trange(), model.similarity(sim.data, cue))
plt.legend(model.get_output_vocab('cue').keys, fontsize='x-small')
plt.ylabel("cue")

plt.subplot(5, 1, 4)
for pointer in ['RED * CIRCLE', 'BLUE * SQUARE']:
    plt.plot(
        sim.trange(),
        vocab.parse(pointer).dot(sim.data[conv].T),
        label=pointer)
plt.legend(fontsize='x-small')
plt.ylabel("convolved")

plt.subplot(5, 1, 5)
plt.plot(sim.trange(), spa.similarity(sim.data[out], vocab))
plt.legend(model.get_output_vocab('out').keys, fontsize='x-small')
plt.ylabel("output")
plt.xlabel("time [s]");


# The last plot shows that the output is most similar to the semantic pointer bound to the current cue. For example, when `RED` and `CIRCLE` are being convolved and the cue is `CIRCLE`, the output is most similar to `RED`.

plt.show()
