
# coding: utf-8

## Nengo example: Routed sequencing

# This demo uses the basal ganglia model to cycle through a 5 element sequence, where an arbitrary start can be presented to the model. The addition of routing allows the system to choose between two different actions: whether to go through the sequence, or be driven by the visual input. If the visual input has its value set to 0.8*START+D (for instance), it will begin cycling through at D->E, etc. The 0.8 scaling helps ensure start is unlikely to accidently match other SPAs (which can be a problem in low dimensional examples like this one).

# In[ ]:

# Setup for the notebook
import matplotlib.pyplot as plt

import nengo
from nengo import spa


### Step 1: Create the model

# Notice that when you specify actions, you're determining which modules are connected to which. For example, by having a mapping that depends on the state of cortex, you are determining that the cortex and basal ganglia must be connected. As well, when you specify that the result of the action changes the state of cortex, then you are determining that thalamus must be connected to cortex.
# 

# In[ ]:

# Number of dimensions for the Semantic Pointers
dimensions = 16

# Make a model object with the SPA network
model = spa.SPA(label='Routed_Sequence')

with model:
    # Specify the modules to be used
    model.cortex = spa.State(dimensions=dimensions)
    model.vision = spa.State(dimensions=dimensions)
    # Specify the action mapping
    actions = spa.Actions(
        'dot(vision, START) --> cortex = vision',
        'dot(cortex, A) --> cortex = B',
        'dot(cortex, B) --> cortex = C',
        'dot(cortex, C) --> cortex = D',
        'dot(cortex, D) --> cortex = E',
        'dot(cortex, E) --> cortex = A'
    )
    model.bg = spa.BasalGanglia(actions=actions)
    model.thal = spa.Thalamus(model.bg)


### Step 2: Provide the input

# Specify a function that provides the model with an initial input semantic pointer.

# In[ ]:

def start(t):
    if t < 0.1:
        return '0.8*START+D'
    else:
        return '0'


with model:
    model.input = spa.Input(vision=start)


### Step 3: Probe the output

# In[ ]:

with model:
    cortex = nengo.Probe(model.cortex.output, synapse=0.01)
    vision = nengo.Probe(model.vision.output, synapse=0.01)
    actions = nengo.Probe(model.thal.actions.output, synapse=0.01)
    utility = nengo.Probe(model.bg.input, synapse=0.01)


### Step 4: Run the model

# In[ ]:

# Create the simulator object
with nengo.Simulator(model) as sim:
    # Simulate the model for 0.5 seconds
    sim.run(0.5)


### Step 5: Plot the results

# In[ ]:

fig = plt.figure(figsize=(12, 8))
p1 = fig.add_subplot(4, 1, 1)
p1.plot(sim.trange(), model.similarity(sim.data, vision))
p1.legend(model.get_output_vocab('vision').keys, fontsize='x-small')
p1.set_ylabel('Vision')

p2 = fig.add_subplot(4, 1, 2)
p2.plot(sim.trange(), model.similarity(sim.data, cortex))
p2.legend(model.get_output_vocab('cortex').keys, fontsize='x-small')
p2.set_ylabel('Cortex')

p3 = fig.add_subplot(4, 1, 3)
p3.plot(sim.trange(), sim.data[actions])
p3.set_ylabel('Action')

p4 = fig.add_subplot(4, 1, 4)
p4.plot(sim.trange(), sim.data[utility])
p4.set_ylabel('Utility')

fig.subplots_adjust(hspace=0.2)


plt.show()
