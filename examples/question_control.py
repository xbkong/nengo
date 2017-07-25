
# coding: utf-8

# # Nengo example: Controlled question answering
# 
# This demo implements a simple form of question answering by using the basal ganglia to store and retrieve information from working memory in response to visual input. More specifically, the basal ganglia decides what to do with the information in the visual channel based on its content (i.e. whether it is a statement or a question)
# 
# When you run the network, it will start by binding `RED` and `CIRCLE` and then binding `BLUE` and `SQUARE` so the memory essentially has `RED * CIRCLE + BLUE * SQUARE`. It does this because it is told that `RED * CIRCLE` is a STATEMENT (i.e. `RED * CIRCLE + STATEMENT` in the code) as is `BLUE * SQUARE`. Then it is presented with something like `QUESTION + RED` (i.e., "What is red?"). The basal ganglia then reroutes that input to be compared to what is in working memory and the result shows up in the motor channel.

# In[ ]:

# Setup for the notebook
import matplotlib.pyplot as plt

import nengo
from nengo import spa


# ## Create the model
# 
# Notice that when you specify actions, you're determining which modules are connected to which. For example, by having a mapping that depends on the state of cortex, you are determining that the cortex and basal ganglia must be connected. As well, when you specify that the result of the action changes the state of cortex, then you are determining that thalamus must be connected to cortex.

# In[ ]:

# Number of dimensions for the Semantic Pointers
dimensions = 128

# Make a model object with the SPA network
model = spa.SPA(label='Controlled_Question_Answering')

with model:
    # Specify the modules to be used
    model.vision = spa.State(dimensions=dimensions, neurons_per_dimension=100) 
    model.motor = spa.State(dimensions=dimensions, neurons_per_dimension=100)
    # Set the feedback connection on the population to 1 to create a memory
    model.memory = spa.State(
        dimensions=dimensions, neurons_per_dimension=100, feedback=1)

    # Specify the action mapping
    actions = spa.Actions(
        'dot(vision, STATEMENT) --> memory = vision - STATEMENT',
        'dot(vision, QUESTION) --> motor = ~vision*memory',
    )
    model.bg = spa.BasalGanglia(actions=actions)
    model.thal = spa.Thalamus(model.bg)


# ## Provide the input

# In[ ]:

def Input(t):
    if 0.1 < t < 0.3:
        return 'STATEMENT+RED*CIRCLE'
    elif 0.35 < t < 0.5:
        return 'STATEMENT+BLUE*SQUARE'
    elif 0.55 < t < 0.7:
        return 'QUESTION+RED'
    elif 0.75 < t < 0.9:
        return 'QUESTION+BLUE'
    else:
        return '0'


with model:
    model.input = spa.Input(vision=Input)


# ## Probe the output

# In[ ]:

with model:
    vision = nengo.Probe(model.vision.output, synapse=0.03)
    motor = nengo.Probe(model.motor.output, synapse=0.03)
    memory = nengo.Probe(model.memory.output, synapse=0.03)
    actions = nengo.Probe(model.thal.actions.output, synapse=0.01)
    utility = nengo.Probe(model.bg.input, synapse=0.01)


# ## Run the model

# In[ ]:

# Create the simulator object
with nengo.Simulator(model) as sim:
    # Simulate the model for 1.2 seconds
    sim.run(1.2)


# ## Plot the results

# In[ ]:

fig = plt.figure(figsize=(12, 8))
p1 = fig.add_subplot(5, 1, 1)
p1.plot(sim.trange(), model.similarity(sim.data, vision))
p1.legend(model.get_output_vocab('vision').keys, fontsize='x-small')
p1.set_ylabel('vision')

p2 = fig.add_subplot(5, 1, 2)
p2.plot(sim.trange(), model.similarity(sim.data, memory))
p2.legend(model.get_output_vocab('memory').keys, fontsize='x-small')
p2.set_ylabel('memory')

p3 = fig.add_subplot(5, 1, 3)
p3.plot(sim.trange(), model.similarity(sim.data, motor))
p3.legend(model.get_output_vocab('motor').keys, fontsize='x-small')
p3.set_ylabel('motor')

p4 = fig.add_subplot(5, 1, 4)
p4.plot(sim.trange(), sim.data[actions])
p4.set_ylabel('action')

p5 = fig.add_subplot(5, 1, 5)
p5.plot(sim.trange(), sim.data[utility])
p5.set_ylabel('utility')
p5.set_xlabel('time [s]')

fig.subplots_adjust(hspace=0.2)


plt.show()
