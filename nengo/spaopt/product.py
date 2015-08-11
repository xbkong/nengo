import numpy as np

import nengo
from nengo.networks.ensemblearray import EnsembleArray
from nengo.utils.compat import is_number
from nengo.dists import Choice
from nengo.utils.optimization import SubvectorRadiusOptimizer


# TODO unittest pure product
class Product(nengo.Network):
    """Computes the element-wise product of two (scaled) unit vectors.

    Requires Scipy.
    """

    def __init__(self, n_neurons, dimensions, radius=1.0):
        super(Product, self).__init__(self)

        with self:
            self.A = nengo.Node(size_in=dimensions, label="A")
            self.B = nengo.Node(size_in=dimensions, label="B")
            self.dimensions = dimensions

            optimizer = SubvectorRadiusOptimizer(n_neurons, 2)
            scaled_r = radius * optimizer.find_optimal_radius(dimensions, 1)

            self.product = nengo.networks.Product(
                n_neurons, dimensions, input_magnitude=scaled_r / np.sqrt(2.))

            nengo.Connection(self.A, self.product.A, synapse=None)
            nengo.Connection(self.B, self.product.B, synapse=None)

            self.output = self.product.output

    def dot_product_transform(self, scale=1.0):
        """Returns a transform for output to compute the scaled dot product."""
        return scale * np.ones((1, self.dimensions))
