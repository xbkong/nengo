import pytest

import nengo
from nengo.exceptions import NetworkContextError
from nengo.params import params



class CopyTest(object):
    def create_original(self):
        raise NotImplementedError()

    def test_copy_in_network(self):
        original = self.create_original()

        with nengo.Network() as model:
            copy = original.copy()
        assert copy in model.all_objects

        self.assert_is_copy(copy, original)

    def test_copy_in_network_without_adding(self):
        original = self.create_original()

        with nengo.Network() as model:
            copy = original.copy(add_to_container=False)
        assert copy not in model.all_objects

        self.assert_is_copy(copy, original)

    def test_copy_outside_network(self):
        original = self.create_original()
        with pytest.raises(NetworkContextError):
            original.copy()

    def test_copy_outside_network_without_adding(self):
        original = self.create_original()
        copy = original.copy(add_to_container=False)
        self.assert_is_copy(copy, original)

    @staticmethod
    def assert_is_copy(copy, original):
        assert copy is not original  # ensures separate parameters
        for param in params(copy):
            assert getattr(copy, param) == getattr(original, param)


class TestCopyEnsemble(CopyTest):
    def create_original(self):
        with nengo.Network() as _:
            e = nengo.Ensemble(10, 1, radius=2.)
        return e

    def test_neurons_reference_copy(self):
        original = self.create_original()
        copy = original.copy(add_to_container=False)
        assert original.neurons.ensemble is original
        assert copy.neurons.ensemble is copy


# Probe
# Ensemble
# Node
# Connection

# Network

# Process
# Learning rule
# Synapse


# copy, copy with add to network, pickle and unpickle
