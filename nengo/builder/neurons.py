import numpy as np

from nengo.builder.builder import Builder
from nengo.builder.signal import Signal
from nengo.builder.operator import Operator
from nengo.neurons import (AdaptiveLIF, AdaptiveLIFRate, Izhikevich, LIF,
                           LIFRate, RectifiedLinear, Sigmoid)


class SimNeurons(Operator):
    """Set output to neuron model output for the given input current."""

    def __init__(self, neurons, J, output, states=[], params=[], tag=None):
        self.neurons = neurons
        self.J = J
        self.output = output
        self.states = states
        self.params = params
        self.tag = tag

        self.sets = [output] + states
        self.incs = []
        self.reads = [J]
        self.updates = []

    def __str__(self):
        return "SimNeurons(%s, %s, %s, %s%s)" % (
            self.neurons, self.J, self.output, self.states, self._tagstr)

    def make_step(self, signals, dt, rng):
        J = signals[self.J]
        output = signals[self.output]
        states = [signals[state] for state in self.states]
        args = states + list(self.params)

        def step_simneurons():
            self.neurons.step_math(dt, J, output, *args)
        return step_simneurons


@Builder.register(RectifiedLinear)
def build_rectifiedlinear(model, reclinear, neurons, rng=None):
    model.add_op(SimNeurons(neurons=reclinear,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out']))


@Builder.register(Sigmoid)
def build_sigmoid(model, sigmoid, neurons, rng=None):
    model.add_op(SimNeurons(neurons=sigmoid,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out']))


@Builder.register(LIFRate)
def build_lifrate(model, lifrate, neurons, rng=None):
    model.add_op(SimNeurons(neurons=lifrate,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out']))


@Builder.register(LIF)
def build_lif(model, lif, neurons, rng=None):
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
def build_alifrate(model, alifrate, neurons, rng=None):
    model.sig[neurons]['adaptation'] = Signal(
        np.zeros(neurons.size_in), name="%s.adaptation" % neurons)
    model.add_op(SimNeurons(neurons=alifrate,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out'],
                            states=[model.sig[neurons]['adaptation']]))


@Builder.register(AdaptiveLIF)
def build_alif(model, alif, neurons, rng=None):
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
def build_izhikevich(model, izhikevich, neurons, rng=None):
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


from nengo.neurons import EIF
@Builder.register(EIF)
def build_eif(model, eif, neurons, rng=None):
    n = neurons.size_in
    t_ref = eif.t_ref.sample(n, rng=rng)
    params = eif.params.sample(n, d=5, rng=rng)
    C, tau, E, VT, DT = params.T
    model.sig[neurons]['V'] = Signal(E * np.ones(n), name="%s.V" % neurons)
    model.sig[neurons]['W'] = Signal(np.zeros(n), name="%s.W" % neurons)
    model.add_op(SimNeurons(neurons=eif,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out'],
                            states=[model.sig[neurons]['V'],
                                    model.sig[neurons]['W']],
                            params=[t_ref, C, tau, E, VT, DT]))
