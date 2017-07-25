
# coding: utf-8

# # Nengo example: Simple question answering with memory
# 
# This demo implements a simple form of question answering. Two features (color and shape) will be bound by circular convolution and stored in a memory population. A cue will be used to determine either one of the features by deconvolution.
# 
# When you run the network, it will start by binding `RED` and `CIRCLE` for 0.25 seconds and then binding `BLUE` and `SQUARE` for 0.25 seconds. Both bound semantic pointers are stored in a memory population. Then the network is asked with the cue. For example, when the cue is `CIRCLE` the network will respond with `RED`.

# In[ ]:

import matplotlib.pyplot as plt

import nengo
from nengo import spa


# ## Create the model

# In[ ]:

# Number of dimensions for the Semantic Pointers
dimensions = 32

model = spa.SPA(label="Simple question answering")

with model:
    model.color_in = spa.State(dimensions=dimensions)
    model.shape_in = spa.State(dimensions=dimensions)
    model.conv = spa.State(dimensions=dimensions,
                           neurons_per_dimension=100,
                           feedback=1,
                           feedback_synapse=0.4)
    model.cue = spa.State(dimensions=dimensions)
    model.out = spa.State(dimensions=dimensions)

    # Connect the state populations
    cortical_actions = spa.Actions(
        'conv = color_in * shape_in',
        'out = conv * ~cue'
    )
    model.cortical = spa.Cortical(cortical_actions)  


# ## Provide the input
# 
# The color input will `RED` and then `BLUE` for 0.25 seconds each before being turned off. In the same way the shape input will be `CIRCLE` and then `SQUARE` for 0.25 seconds each. Thus, the network will bind alternatingly `RED * CIRCLE` and `BLUE * SQUARE` for 0.5 seconds each.
# 
# The cue for deconvolving bound semantic pointers will be turned off for 0.5 seconds and then cycles through `CIRCLE`, `RED`, `SQUARE`, and `BLUE` within one second. 

# In[ ]:

def color_input(t):
    if t < 0.25:
        return 'RED'
    elif t < 0.5:
        return 'BLUE'
    else:
        return '0'


def shape_input(t):
    if t < 0.25:
        return 'CIRCLE'
    elif t < 0.5:
        return 'SQUARE'
    else:
        return '0'


def cue_input(t):
    if t < 0.5:
        return '0'
    sequence = ['0', 'CIRCLE', 'RED', '0', 'SQUARE', 'BLUE']
    idx = int(((t - 0.5) // (1. / len(sequence))) % len(sequence))
    return sequence[idx]


with model:
    model.inp = spa.Input(
        color_in=color_input, shape_in=shape_input, cue=cue_input)


# ## Probe the output

# In[ ]:

with model:
    model.config[nengo.Probe].synapse = nengo.Lowpass(0.03)
    color_in = nengo.Probe(model.color_in.output)
    shape_in = nengo.Probe(model.shape_in.output)
    cue = nengo.Probe(model.cue.output)
    conv = nengo.Probe(model.conv.output)
    out = nengo.Probe(model.out.output)


# ## Run the model

# In[ ]:

with nengo.Simulator(model) as sim:
    sim.run(3.)


# ## Plot the results

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
plt.ylabel("Output")
plt.xlabel("time [s]")


# The plots of `shape`, `color`, and `convolved` show that first `RED * CIRCLE` and then `BLUE * SQUARE` will be loaded into the `convolved` buffer so after 0.5 seconds it represents the superposition `RED * CIRCLE + BLUE * SQUARE`.
# 
# The last plot shows that the output is most similar to the semantic pointer bound to the current cue. For example, when `RED` and `CIRCLE` are being convolved and the cue is `CIRCLE`, the output is most similar to `RED`. Thus, it is possible to unbind semantic pointers from the superposition stored in `convolved`.

plt.show()
