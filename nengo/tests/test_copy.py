import numpy as np
import pytest

import nengo
from nengo.exceptions import NetworkContextError
from nengo.params import params
from nengo.utils.compat import pickle


def assert_is_copy(copy, original):
    assert copy is not original  # ensures separate parameters
    for param in params(copy):
        if isinstance(getattr(copy, param), nengo.solvers.Solver):
            continue  # FIXME
        if isinstance(getattr(copy, param), nengo.base.NengoObject):
            continue  # FIXME
        assert getattr(copy, param) == getattr(original, param)


class CopyTest(object):
    def create_original(self):
        raise NotImplementedError()

    def test_copy_in_network(self):
        original = self.create_original()

        with nengo.Network() as model:
            copy = original.copy()
        assert copy in model.all_objects

        assert_is_copy(copy, original)

    def test_copy_in_network_without_adding(self):
        original = self.create_original()

        with nengo.Network() as model:
            copy = original.copy(add_to_container=False)
        assert copy not in model.all_objects

        assert_is_copy(copy, original)

    def test_copy_outside_network(self):
        original = self.create_original()
        with pytest.raises(NetworkContextError):
            original.copy()

    def test_copy_outside_network_without_adding(self):
        original = self.create_original()
        copy = original.copy(add_to_container=False)
        assert_is_copy(copy, original)


class PickleTest(object):
    def create_original(self):
        raise NotImplementedError()

    def test_pickle_roundtrip(self):
        original = self.create_original()
        copy = pickle.loads(pickle.dumps(original))
        assert_is_copy(copy, original)


class TestCopyEnsemble(CopyTest, PickleTest):
    def create_original(self):
        with nengo.Network():
            e = nengo.Ensemble(10, 1, radius=2.)
        return e

    def test_neurons_reference_copy(self):
        original = self.create_original()
        copy = original.copy(add_to_container=False)
        assert original.neurons.ensemble is original
        assert copy.neurons.ensemble is copy


class TestCopyProbe(CopyTest, PickleTest):
    def create_original(self):
        with nengo.Network():
            e = nengo.Ensemble(10, 1)
            p = nengo.Probe(e, synapse=0.01)
        return p


class TestCopyNode(CopyTest, PickleTest):
    def create_original(self):
        with nengo.Network():
            n = nengo.Node(np.min, size_in=2, size_out=2)
        return n

class TestCopyConnection(CopyTest, PickleTest):
    def create_original(self):
        with nengo.Network(self):
            e1 = nengo.Ensemble(10, 1)
            e2 = nengo.Ensemble(10, 1)
            c = nengo.Connection(e1, e2, transform=2.)
        return c

# Network

# Process
# Learning rule
# Synapse


# copy, copy with add to network, pickle and unpickle
