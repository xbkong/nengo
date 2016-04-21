import numpy as np

from nengo.builder.builder import Builder
from nengo.builder.signal import Signal
from nengo.builder.operator import Operator
from nengo.neurons import (AdaptiveLIF, AdaptiveLIFRate, Izhikevich, LIF,
                           LIFRate, RectifiedLinear, Sigmoid)


class SimNeurons(Operator):
    """Set output to neuron model output for the given input current."""

    def __init__(self, neurons, J, output, states=[], tag=None):
        self.neurons = neurons
        self.J = J
        self.output = output
        self.states = states
        self.tag = tag

        self.sets = [output] + states
        self.incs = []
        self.reads = [J]
        self.updates = []

    def _descstr(self):
        return '%s, %s, %s' % (self.neurons, self.J, self.output)

    def make_step(self, signals, dt, rng):
        J = signals[self.J]
        output = signals[self.output]
        states = [signals[state] for state in self.states]

        def step_simneurons():
            self.neurons.step_math(dt, J, output, *states)
        return step_simneurons


@Builder.register(RectifiedLinear)
def build_rectifiedlinear(model, reclinear, neurons):
    model.add_op(SimNeurons(neurons=reclinear,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out']))


@Builder.register(Sigmoid)
def build_sigmoid(model, sigmoid, neurons):
    model.add_op(SimNeurons(neurons=sigmoid,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out']))


@Builder.register(LIFRate)
def build_lifrate(model, lifrate, neurons):
    model.add_op(SimNeurons(neurons=lifrate,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out']))


@Builder.register(LIF)
def build_lif(model, lif, neurons):
    """Builds a `.LIF` object into a model.

    In addition to adding a `.SimNeurons` operator, this build function sets up
    signals to track the voltage and refractory times for each neuron.

    Parameters
    ----------
    model : Model
        The model to build into.
    lif : LIF
        Neuron type to build.
    neuron : Neurons
        The neuron population object corresponding to the neuron type.

    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.LIF` instance.
    """

    model.sig[neurons]['voltage'] = Signal(
        np.zeros(neurons.size_in), name="%s.voltage" % neurons)
    model.sig[neurons]['refractory_time'] = Signal(
        np.zeros(neurons.size_in), name="%s.refractory_time" % neurons)
    model.add_op(SimNeurons(
        neurons=lif,
        J=model.sig[neurons]['in'],
        output=model.sig[neurons]['out'],
        states=[model.sig[neurons]['voltage'],
                model.sig[neurons]['refractory_time']]))


@Builder.register(AdaptiveLIFRate)
def build_alifrate(model, alifrate, neurons):
    """Builds an `.AdaptiveLIFRate` object into a model.

    In addition to adding a `.SimNeurons` operator, this build function sets up
    signals to track the adaptation term for each neuron.

    Parameters
    ----------
    model : Model
        The model to build into.
    alifrate : AdaptiveLIFRate
        Neuron type to build.
    neuron : Neurons
        The neuron population object corresponding to the neuron type.

    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.AdaptiveLIFRate` instance.
    """

    model.sig[neurons]['adaptation'] = Signal(
        np.zeros(neurons.size_in), name="%s.adaptation" % neurons)
    model.add_op(SimNeurons(neurons=alifrate,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out'],
                            states=[model.sig[neurons]['adaptation']]))


@Builder.register(AdaptiveLIF)
def build_alif(model, alif, neurons):
    """Builds an `.AdaptiveLIF` object into a model.

    In addition to adding a `.SimNeurons` operator, this build function sets up
    signals to track the voltage, refractory time, and adaptation term
    for each neuron.

    Parameters
    ----------
    model : Model
        The model to build into.
    alif : AdaptiveLIF
        Neuron type to build.
    neuron : Neurons
        The neuron population object corresponding to the neuron type.

    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.AdaptiveLIF` instance.
    """

    model.sig[neurons]['voltage'] = Signal(
        np.zeros(neurons.size_in), name="%s.voltage" % neurons)
    model.sig[neurons]['refractory_time'] = Signal(
        np.zeros(neurons.size_in), name="%s.refractory_time" % neurons)
    model.sig[neurons]['adaptation'] = Signal(
        np.zeros(neurons.size_in), name="%s.adaptation" % neurons)
    model.add_op(SimNeurons(neurons=alif,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out'],
                            states=[model.sig[neurons]['voltage'],
                                    model.sig[neurons]['refractory_time'],
                                    model.sig[neurons]['adaptation']]))


@Builder.register(Izhikevich)
def build_izhikevich(model, izhikevich, neurons):
    """Builds an `.Izhikevich` object into a model.

    In addition to adding a `.SimNeurons` operator, this build function sets up
    signals to track the voltage and recovery terms for each neuron.

    Parameters
    ----------
    model : Model
        The model to build into.
    izhikevich : Izhikevich
        Neuron type to build.
    neuron : Neurons
        The neuron population object corresponding to the neuron type.

    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.Izhikevich` instance.
    """

    model.sig[neurons]['voltage'] = Signal(
        np.ones(neurons.size_in) * izhikevich.reset_voltage,
        name="%s.voltage" % neurons)
    model.sig[neurons]['recovery'] = Signal(
        np.ones(neurons.size_in)
        * izhikevich.reset_voltage
        * izhikevich.coupling, name="%s.recovery" % neurons)
    model.add_op(SimNeurons(neurons=izhikevich,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out'],
                            states=[model.sig[neurons]['voltage'],
                                    model.sig[neurons]['recovery']]))
