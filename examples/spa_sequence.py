
# coding: utf-8

## Nengo example: Routing through a sequence

# This demo uses the basal ganglia model to cycle through a sequence of five representations. The model incorporates a memory, which allows the basal ganglia to update that memory based on a set of condition/action mappings. The mappings are defined in the code such that A->B, B->C, etc. until E->A completing a loop. 
# 
# When you run this demo, the sequence will be repeated indefinitely. It is interesting to note the time between the ‘peaks’ of the selected items. It's about 40ms for this simple action.

# In[ ]:

# Setup for the notebook
import matplotlib.pyplot as plt

import nengo
from nengo import spa


### Step 1: Create the model

# Notice that when you specify actions, you're determining which modules are connected to which.  For example, by having a mapping that depends on the state of cortex, you are determining that the cortex and basal ganglia must be connected.  As well, when you specify that the result of the action changes the state of cortex, then you are determining that thalamus must be connected to cortex.

# In[ ]:

# Number of dimensions for the Semantic Pointers
dimensions = 16

# Make a model object with the SPA network
model = spa.SPA(label='Sequence_Module')

with model:
    # Specify the modules to be used
    model.cortex = spa.State(dimensions=dimensions)
    # Specify the action mapping
    actions = spa.Actions(
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
    if t < 0.05:
        return 'A'
    else:
        return '0'


with model:
    model.input = spa.Input(cortex=start)


### Step 3: Probe the output

# In[ ]:

with model:
    cortex = nengo.Probe(model.cortex.output, synapse=0.01)
    actions = nengo.Probe(model.thal.actions.output, synapse=0.01)
    utility = nengo.Probe(model.bg.input, synapse=0.01)


### Step 4: Run the model

# In[ ]:

# Create the simulator object
with nengo.Simulator(model) as sim:
    # Simulate the model for 0.5 seconds
    sim.run(0.5)


### Step 5: Plot the results

# Plot the results of the simulation. The first figure shows the semantic pointer representation of the values stored in the "cortex" module. The second figure shows the actions being executed, and the third figure shows the utility (similarity) of the cortex representation to the conditions of each action.

# In[ ]:

fig = plt.figure(figsize=(12, 8))
p1 = fig.add_subplot(3, 1, 1)

p1.plot(sim.trange(), model.similarity(sim.data, cortex))
p1.legend(model.get_output_vocab('cortex').keys, fontsize='x-small')
p1.set_ylabel('State')

p2 = fig.add_subplot(3, 1, 2)
p2.plot(sim.trange(), sim.data[actions])
p2_legend_txt = [a.effect for a in model.bg.actions.actions]
p2.legend(p2_legend_txt, fontsize='x-small')
p2.set_ylabel('Action')

p3 = fig.add_subplot(3, 1, 3)
p3.plot(sim.trange(), sim.data[utility])
p3_legend_txt = [a.condition for a in model.bg.actions.actions]
p3.legend(p3_legend_txt, fontsize='x-small')
p3.set_ylabel('Utility')

fig.subplots_adjust(hspace=0.2)


plt.show()
