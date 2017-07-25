
# coding: utf-8

## Nengo example: Parsing simple commands

# This example is a simplified version
# of the language parsing model presented in
# [Stewart & Eliasmith, 2013](http://compneuro.uwaterloo.ca/publications/stewart2013.html).
# Please refer to that paper for the high-level details.

# In[ ]:

# Setup for the notebook
import matplotlib.pyplot as plt

import nengo
from nengo import spa


### Step 1: Create the model

# In[ ]:

# Number of dimensions for the SPs
dimensions = 64

# Make a model object with the SPA network
model = spa.SPA(label='Parser')

with model:
    # Specify the modules to be used
    model.vision = spa.State(dimensions=dimensions, neurons_per_dimension=100)
    model.phrase = spa.State(dimensions=dimensions, neurons_per_dimension=100)
    model.motor = spa.State(dimensions=dimensions, neurons_per_dimension=100)
    model.noun = spa.State(dimensions=dimensions,
                           neurons_per_dimension=100,
                           feedback=1.0)
    model.verb = spa.State(dimensions=dimensions,
                           neurons_per_dimension=100,
                           feedback=1.0)

    # Specify the action mapping
    actions = spa.Actions(
        'dot(vision, WRITE) --> verb=vision',
        'dot(vision, ONE+TWO+THREE) --> noun=vision',
        '0.5*(dot(NONE-WRITE-ONE-TWO-THREE, vision) '
        '+ dot(phrase, WRITE*VERB)) '
        '--> motor=phrase*~NOUN',
    )
    cortical_actions = spa.Actions(
            'phrase=noun*NOUN + verb*VERB',
    )
    model.bg = spa.BasalGanglia(actions=actions)
    model.thal = spa.Thalamus(model.bg)
    model.cortical = spa.Cortical(actions=cortical_actions)


### Step 2: Provide the input

# In[ ]:

def input_vision(t):
    sequence = 'WRITE ONE NONE WRITE TWO NONE THREE WRITE NONE'.split()
    index = int(t / 0.5) % len(sequence)
    return sequence[index]


with model:
    model.input = spa.Input(vision=input_vision)


### Step 3: Probe the output

# In[ ]:

with model:
    vision = nengo.Probe(model.vision.output, synapse=0.03)
    phrase = nengo.Probe(model.phrase.output, synapse=0.03)
    motor = nengo.Probe(model.motor.output, synapse=0.03)
    noun = nengo.Probe(model.noun.output, synapse=0.03)
    verb = nengo.Probe(model.verb.output, synapse=0.03)
    actions = nengo.Probe(model.thal.actions.output, synapse=0.01)
    utility = nengo.Probe(model.bg.input, synapse=0.01)


### Step 4: Run the model

# In[ ]:

# Create the simulator object
with nengo.Simulator(model) as sim:
    # Simulate the model for 4.5 seconds
    sim.run(4.5)


### Step 5: Plot the results

# In[ ]:

fig = plt.figure(figsize=(16, 12))
p1 = fig.add_subplot(7, 1, 1)
p1.plot(sim.trange(), model.similarity(sim.data, vision))
p1.legend(model.get_output_vocab('vision').keys, fontsize='x-small')
p1.set_ylabel('Vision')

p2 = fig.add_subplot(7, 1, 2)
p2.plot(sim.trange(), model.similarity(sim.data, phrase))
p2.legend(model.get_output_vocab('phrase').keys, fontsize='x-small')
p2.set_ylabel('Phrase')

p3 = fig.add_subplot(7, 1, 3)
p3.plot(sim.trange(), model.similarity(sim.data, motor))
p3.legend(model.get_output_vocab('motor').keys, fontsize='x-small')
p3.set_ylabel('Motor')

p4 = fig.add_subplot(7, 1, 4)
p4.plot(sim.trange(), model.similarity(sim.data, noun))
p4.legend(model.get_output_vocab('noun').keys, fontsize='x-small')
p4.set_ylabel('Noun')

p5 = fig.add_subplot(7, 1, 5)
p5.plot(sim.trange(), model.similarity(sim.data, verb))
p5.legend(model.get_output_vocab('verb').keys, fontsize='x-small')
p5.set_ylabel('Verb')

p6 = fig.add_subplot(7, 1, 6)
p6.plot(sim.trange(), sim.data[actions])
p6.set_ylabel('Action')

p7 = fig.add_subplot(7, 1, 7)
p7.plot(sim.trange(), sim.data[utility])
p7.set_ylabel('Utility')

fig.subplots_adjust(hspace=0.2)


plt.show()
