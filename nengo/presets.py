"""Configuration presets for common use cases."""

import nengo


def ThresholdingPreset(threshold):
    """Configuration preset for a thresholding ensemble.

    This preset adjust ensemble parameters for thresholding. The ensemble
    neurons will only fire for values above the threshold. One can either
    decode the represented value (if it is above the threshold) or decode a
    step function if a binary classification is desired.

    This preset sets:

    - The intercepts to be between `threshold` and 1 with an exponential
      distribution (shape parameter of 0.15). This clusters intercepts near
      the threshold for a better approximation.
    - The encoders to 1.
    - The dimensions to 1.
    - The evaluation points to be between `threshold` and 1. with a uniform
      distribution.

    Parameters
    ----------
    threshold : float
        Threshold of ensembles using this configuration preset.

    Returns
    -------
    :class:`nengo.Config`
        Configuration with presets.
    """
    config = nengo.Config(nengo.Ensemble)
    config[nengo.Ensemble].dimensions = 1
    config[nengo.Ensemble].intercepts = nengo.dists.Exponential(
        0.15, threshold, 1.)
    config[nengo.Ensemble].encoders = nengo.dists.Choice([[1]])
    config[nengo.Ensemble].eval_points = nengo.dists.Uniform(threshold, 1.)
    return config
