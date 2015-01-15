import logging

import pytest

import nengo
from nengo.synapses import SynapseParam
from nengo.utils.functions import whitenoise
from nengo.utils.numpy import filt, lti, alpha_transfer_fn
from nengo.utils.testing import allclose

logger = logging.getLogger(__name__)


def run_synapse(Simulator, synapse, dt=1e-3, runtime=1., n_neurons=None):
    model = nengo.Network(seed=2984)
    with model:
        u = nengo.Node(output=whitenoise(0.1, 5, seed=328))

        if n_neurons is not None:
            a = nengo.Ensemble(n_neurons, 1)
            nengo.Connection(u, a, synapse=None)
            target = a
        else:
            target = u

        ref = nengo.Probe(target)
        filtered = nengo.Probe(target, synapse=synapse)

    sim = Simulator(model, dt=dt)
    sim.run(runtime)

    return sim.trange(), sim.data[ref], sim.data[filtered]


def test_lowpass(Simulator, plt):
    dt = 1e-3
    tau = 0.03

    t, x, yhat = run_synapse(Simulator, nengo.synapses.Lowpass(tau), dt=dt)
    y = filt(x, tau / dt)

    assert allclose(t, y, yhat, delay=dt, plt=plt)


def test_alpha(Simulator, plt):
    dt = 1e-3
    tau = 0.03

    t, x, yhat = run_synapse(Simulator, nengo.synapses.Alpha(tau), dt=dt)

    y = lti(x, alpha_transfer_fn(tau, dt), normalize=False)
    assert allclose(t, y, yhat, delay=dt, atol=5e-6, plt=plt)


def test_decoders(Simulator, nl, plt):
    dt = 1e-3
    tau = 0.01

    t, x, yhat = run_synapse(
        Simulator, nengo.synapses.Lowpass(tau), dt=dt, n_neurons=100)

    y = filt(x, tau / dt)
    assert allclose(t, y, yhat, delay=dt, plt=plt)


@pytest.mark.optional  # the test requires scipy
def test_general(Simulator, plt):
    import scipy.signal

    dt = 1e-3
    order = 4
    tau = 0.03

    num, den = scipy.signal.butter(order, 1. / tau, analog=True)
    num, den = num.real, den.real
    numi, deni, dt = scipy.signal.cont2discrete((num, den), dt)

    t, x, yhat = run_synapse(
        Simulator, nengo.synapses.LinearFilter(num, den), dt=dt)
    y = lti(x, (numi, deni))

    assert allclose(t, y, yhat, plt=plt)


def test_synapseparam():
    """SynapseParam must be a Synapse, and converts numbers to LowPass."""
    class Test(object):
        sp = SynapseParam(default=nengo.Lowpass(0.1))

    inst = Test()
    assert isinstance(inst.sp, nengo.Lowpass)
    assert inst.sp.tau == 0.1
    # Number are converted to LowPass
    inst.sp = 0.05
    assert isinstance(inst.sp, nengo.Lowpass)
    assert inst.sp.tau == 0.05
    # None has meaning
    inst.sp = None
    assert inst.sp is None
    # Non-synapse not OK
    with pytest.raises(ValueError):
        inst.sp = 'a'


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
