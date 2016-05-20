import numpy as np

from nengo.builder import Builder, Signal
from nengo.builder.connection import BuiltConnection
from nengo.builder.operator import Reset
from nengo.connection import Connection
from nengo.exceptions import BuildError
from nengo.utils.compat import iteritems


class BuiltProbe(object):
    """Builds a `.Probe` object into a model.

    Under the hood, there are two types of probes:
    connection probes and signal probes.

    Connection probes are those that are built by creating a new `.Connection`
    object from the probe's target to the probe, and calling that connection's
    build function. Creating and building a connection ensure that the result
    of probing the target's attribute is the same as would result from that
    target being connected to another object.

    Signal probes are those that are built by finding the correct `.Signal`
    in the model and calling the build function corresponding to the probe's
    synapse.

    Parameters
    ----------
    model : Model
        The model to build into.
    probe : Probe
        The connection to build.

    Notes
    -----
    Sets ``model.params[probe]`` to a list.
    `.Simulator` appends to that list when running a simulation.
    """

    probemap = {
        'Ensemble': {'decoded_output': None,
                     'input': 'in'},
        'Neurons': {'output': None,
                    'spikes': None,
                    'rates': None,
                    'input': 'in'},
        'Node': {'output': None},
        'Connection': {'output': 'weighted',
                       'input': 'in'},
        'LearningRule': {},  # make LR signals probeable, no mapping required
    }

    __slots__ = ('attr',
                 'key',
                 'sample_every',
                 'seed',
                 'seeded',
                 'slice',
                 'solver',
                 'synapse',
                 'target',
                 'typename')

    def __init__(self, probe, obj, seed):
        self.seeded = probe.seed is not None
        self.seed = probe.seed if self.seeded else seed

        self.attr = probe.attr
        self.slice = probe.slice
        self.sample_every = probe.sample_every
        self.solver = probe.solver
        self.synapse = probe.synapse
        self.target = probe.target
        self.typename = 'Probe'

        for nengotype, probeables in iteritems(self.probemap):
            if self.obj.typename == nengotype:
                break
        else:
            raise BuildError(
                "Type %r is not probeable" % self.obj.__class__.__name__)

        self.key = (probeables[probe.attr] if probe.attr in probeables
                    else probe.attr)


@Builder.register(BuiltProbe)
def build_probe(model, probe):
    """Builds a `.Probe` object into a model.
    Under the hood, there are two types of probes:
    connection probes and signal probes.
    Connection probes are those that are built by creating a new `.Connection`
    object from the probe's target to the probe, and calling that connection's
    build function. Creating and building a connection ensure that the result
    of probing the target's attribute is the same as would result from that
    target being connected to another object.
    Signal probes are those that are built by finding the correct `.Signal`
    in the model and calling the build function corresponding to the probe's
    synapse.
    Parameters
    ----------
    model : Model
        The model to build into.
    probe : Probe
        The connection to build.
    Notes
    -----
    Sets ``model.params[probe]`` to a list.
    `.Simulator` appends to that list when running a simulation.
    """
    if probe.key is None:
        conn_probe(model, probe)
    else:
        signal_probe(model, probe)


def conn_probe(model, probe):
    # Connection probes create a connection from the target, and probe
    # the resulting signal (used when you want to probe the default
    # output of an object, which may not have a predefined signal)

    conn = Connection(probe.target, probe, synapse=probe.synapse,
                      solver=probe.solver, add_to_container=False)
    if probe.seeded:
        conn.seed = probe.seed
    conn = BuiltConnection(conn, seed=probe.seed)

    # Make a sink signal for the connection
    model.sig[probe]['in'] = Signal(np.zeros(conn.size_out), name=str(probe))
    model.add_op(Reset(model.sig[probe]['in']))

    # Build the connection
    model.build(conn)


def signal_probe(model, probe):
    # Signal probes directly probe a target signal

    try:
        sig = model.sig[probe.obj][probe.key]
    except IndexError:
        raise BuildError(
            "Attribute %r is not probeable on %s." % (probe.key, probe.obj))

    if probe.slice is not None:
        sig = sig[probe.slice]

    if probe.synapse is None:
        model.sig[probe]['in'] = sig
    else:
        model.sig[probe]['in'] = model.build(probe.synapse, sig)
