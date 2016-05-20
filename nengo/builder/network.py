import logging

import numpy as np

import nengo.utils.numpy as npext
from nengo.builder import Builder
from nengo.builder.connection import BuiltConnection
from nengo.builder.ensemble import BuiltEnsemble
from nengo.builder.node import BuiltNode
from nengo.builder.probe import BuiltProbe
from nengo.cache import NoDecoderCache
from nengo.network import Network
from nengo.utils.compat import is_integer

logger = logging.getLogger(__name__)


class BuiltNetwork(object):
    """Built version of a `.Network`.

    Like the `.Network`, this is mainly a container for the built versions
    of other objects. The ``BuiltNetwork`` will create built versions of the
    contents of the original network when it's created.

    The contents of the network are built in the following order:

    1. Ensembles
    2. Nodes
    3. Subnetworks
    4. Connections
    5. Probes

    Before calling any of the individual objects' builders, random number seeds
    are assigned to objects that did not have a seed explicitly set by
    the user. Whether the seed was assigned manually or automatically
    is tracked, and the decoder cache is only used when the seed
    is assigned manually.

    Parameters
    ----------
    network : Network
        The network to build.
    """

    __slots__ = ('_built',
                 '_original',
                 'connections',
                 'decoder_cache',
                 'dt',
                 'ensembles',
                 'label',
                 'networks',
                 'nodes',
                 'probes',
                 'seed',
                 'seeded',
                 'toplevel',
                 'typename')

    def __init__(self, network, dt=0.001, decoder_cache=NoDecoderCache(),
                 toplevel=True, seed=None):
        assert isinstance(network, Network)
        assert toplevel or is_integer(seed), (
            "Only toplevel networks should have seed=None")
        seed = np.random.randint(npext.maxint) if toplevel else seed

        self.typename = type(network).__name__
        self.label = network.label
        self.toplevel = toplevel
        self.seeded = network.seed is not None
        self.seed = network.seed if self.seeded else seed

        # Mappings between built and original and vice versa
        self._built = {}
        self._original = {}

        # Assign seeds to children in a deterministic order
        rng = np.random.RandomState(self.seed)
        seeds = {}
        for obj_type in sorted(network.objects, key=lambda t: t.__name__):
            for obj in network.objects[obj_type]:
                seeds[obj] = rng.randint(npext.maxint)

        # Build objects
        logger.debug("Building ensembles")
        for ens in network.ensembles:
            built = BuiltEnsemble(ens, seeds[ens])
            self.ensembles.append(built)
            self._built[ens] = built
            self._original[built] = ens

        logger.debug("Building nodes")
        for node in network.nodes:
            built = BuiltNode(node, seeds[node])
            self.nodes.append(built)
            self._built[node] = built
            self._original[built] = node

        logger.debug("Building subnetworks")
        for net in network.networks:
            built = BuiltNetwork(net, toplevel=False, seed=seeds[net])
            self.networks.append(built)
            self._built[net] = built
            self._original[built] = net

        logger.debug("Building connections")
        for conn in network.connections:
            built = BuiltConnection(conn, seeds[conn])
            self.connections.append(built)
            self._built[conn] = built
            self._original[built] = conn

        logger.debug("Building probes")
        for probe in network.probes:
            built = BuiltProbe(probe, seeds[probe])
            self.probes.append(built)
            self._built[probe] = built
            self._original[built] = probe

    def get_built(self, obj):
        return self._built[obj]

    def get_original(self, builtobj):
        return self._originals[builtobj]


@Builder.register(BuiltNetwork)
def build_network(model, network):
    """Builds a `.Network` object into a model.
    The network builder does this by mapping each high-level object to its
    associated signals and operators one-by-one, in the following order:
    1. Ensembles, nodes, neurons
    2. Subnetworks (recursively)
    3. Connections, learning rules
    4. Probes
    Before calling any of the individual objects' build functions, random
    number seeds are assigned to objects that did not have a seed explicitly
    set by the user. Whether the seed was assigned manually or automatically
    is tracked, and the decoder cache is only used when the seed is assigned
    manually.
    Parameters
    ----------
    model : Model
        The model to build into.
    network : Network
        The network to build.
    Notes
    -----
    Sets ``model.params[network]`` to ``None``.
    """

    logger.debug("Building ensembles and nodes")
    for obj in network.ensembles + network.nodes:
        model.build(obj)

    logger.debug("Building subnetworks")
    for subnetwork in network.networks:
        model.build(subnetwork)

    logger.debug("Building connections")
    for conn in network.connections:
        # NB: we do these in the order in which they're defined, and build the
        # learning rule in the connection builder. Because learning rules are
        # attached to connections, the connection that contains the learning
        # rule (and the learning rule) are always built *before* a connection
        # that attaches to that learning rule. Therefore, we don't have to
        # worry about connection ordering here.
        # TODO: Except perhaps if the connection being learned
        # is in a subnetwork?
        model.build(conn)

    logger.debug("Building probes")
    for probe in network.probes:
        model.build(probe)
