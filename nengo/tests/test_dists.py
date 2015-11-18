import numpy as np
import pytest

import nengo.dists as dists
import nengo.utils.numpy as npext


def test_pdf(rng):
    s = 0.25
    f = lambda x: (np.exp(-0.5 * (x + 0.5)**2 / s**2) +
                   np.exp(-0.5 * (x - 0.5)**2 / s**2))

    xref = np.linspace(-2, 2, 101)
    pref = f(xref)
    pref /= pref.sum()
    dist = dists.PDF(xref, pref)

    n = 100000
    samples = dist.sample(n, rng=rng)
    h, xedges = np.histogram(samples, bins=101)
    x = 0.5 * (xedges[:-1] + xedges[1:])
    dx = np.diff(xedges)
    y = h / float(h.sum()) / dx
    z = f(x)
    z = z / z.sum() / dx
    assert np.allclose(y, z, atol=0.05)


@pytest.mark.parametrize("low,high", [(-2, -1), (-1, 1), (1, 2), (1, -1)])
def test_uniform(low, high, rng):
    n = 100
    dist = dists.Uniform(low, high)
    samples = dist.sample(n, rng=rng)
    if low < high:
        assert np.all(samples >= low)
        assert np.all(samples < high)
    else:
        assert np.all(samples <= low)
        assert np.all(samples > high)
    hist, _ = np.histogram(samples, bins=5)
    assert np.allclose(hist - np.mean(hist), 0, atol=0.1 * n)


@pytest.mark.parametrize("mean,std", [(0, 1), (0, 0), (10, 2)])
def test_gaussian(mean, std, rng):
    n = 100
    if std <= 0:
        with pytest.raises(ValueError):
            dist = dists.Gaussian(mean, std)
    else:
        dist = dists.Gaussian(mean, std)
        samples = dist.sample(n, rng=rng)
        assert abs(np.mean(samples) - mean) < 2 * std / np.sqrt(n)
        assert abs(np.std(samples) - std) < 0.25  # using chi2 for n=100


@pytest.mark.parametrize("dimensions", [0, 1, 2, 5])
def test_hypersphere(dimensions, rng):
    n = 150 * dimensions
    if dimensions < 1:
        with pytest.raises(ValueError):
            dist = dists.UniformHypersphere().sample(1, dimensions)
    else:
        dist = dists.UniformHypersphere()
        samples = dist.sample(n, dimensions, rng=rng)
        assert samples.shape == (n, dimensions)
        assert np.allclose(np.mean(samples, axis=0), 0, atol=0.1)
        hist, _ = np.histogramdd(samples, bins=5)
        assert np.allclose(hist - np.mean(hist), 0, atol=0.1 * n)


@pytest.mark.parametrize("dimensions", [1, 2, 5])
def test_hypersphere_surface(dimensions, rng):
    n = 150 * dimensions
    dist = dists.UniformHypersphere(surface=True)
    samples = dist.sample(n, dimensions, rng=rng)
    assert samples.shape == (n, dimensions)
    assert np.allclose(npext.norm(samples, axis=1), 1)
    assert np.allclose(np.mean(samples, axis=0), 0, atol=0.25 / dimensions)


@pytest.mark.parametrize("weights", [None, [5, 1, 2, 9], [3, 2, 1, 0]])
def test_choice(weights, rng):
    n = 1000
    choices = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    N = len(choices)

    dist = dists.Choice(choices, weights=weights)
    # If d is passed, it has to match
    with pytest.raises(ValueError):
        dist.sample(n, d=4, rng=rng)
    sample = dist.sample(n, rng=rng)
    tsample, tchoices = list(map(tuple, sample)), list(map(tuple, choices))

    # check that frequency of choices matches weights
    inds = [tchoices.index(s) for s in tsample]
    hist, bins = np.histogram(inds, bins=np.linspace(-0.5, N - 0.5, N + 1))
    p_empirical = hist / float(hist.sum())
    p = np.ones(N) / N if dist.p is None else dist.p
    sterr = 1. / np.sqrt(n)  # expected maximum standard error
    assert np.allclose(p, p_empirical, atol=2 * sterr)


@pytest.mark.parametrize("n,m", [(99, 1), (50, 50)])
def test_sqrt_beta(n, m):
    np.random.seed(33)

    num_samples = 250
    num_bins = 5

    vectors = np.random.randn(num_samples, n + m)
    vectors /= npext.norm(vectors, axis=1, keepdims=True)
    expectation, _ = np.histogram(
        npext.norm(vectors[:, :m], axis=1), bins=num_bins)

    dist = dists.SqrtBeta(n, m)
    samples = dist.sample(num_samples, 1)
    hist, _ = np.histogram(samples, bins=num_bins)

    assert np.all(np.abs(np.asfarray(hist - expectation) / num_samples) < 0.16)


def test_multivariate(rng, plt):
    import scipy.stats as sps

    # std1 = 0.5
    # mean1 = 2
    # lognorm = lambda p: sps.lognorm.ppf(p, std1, scale=np.exp(mean1))

    # low1 = 0
    # high1 = 1
    low1 = 1.5
    high1 = 3
    uniform = lambda p: sps.uniform.ppf(p, scale=high1-low1, loc=low1)

    std2 = 0.5
    mean2 = 2
    norm = lambda p: sps.norm.ppf(p, scale=std2, loc=mean2)

    # R = np.eye(2)
    R = np.array([[1., 0.5], [0.5, 1.]])
    # R = np.array([[1., 0.99], [0.99, 1.]])

    # dist = dists.Multivariate([norm, norm], R)
    # dist = dists.Multivariate([uniform, uniform], R)
    dist = dists.Multivariate([uniform, norm], R)
    # dist = dists.Multivariate([norm, uniform], R)
    x = dist.sample(100000, rng=rng)

    H, xedges, yedges = np.histogram2d(x[:, 0], x[:, 1], bins=30)
    xmids = 0.5 * (xedges[:-1] + xedges[1:])
    ymids = 0.5 * (yedges[:-1] + yedges[1:])

    # check marginals
    xpdf = H.sum(1).astype(float)
    xpdf = xpdf / xpdf.sum()
    assert np.allclose(, sps.norm.pdf(p, scale=std2, loc=mean2), )
    # assert

    # --- plot
    plt.subplot(311)
    plt.contour(ymids, xmids, H)
    plt.subplot(312)

    plt.bar(xmids, H.sum(1))
    # plt.hist(x[:, 0], bins=30)
    plt.subplot(313)
    plt.bar(ymids, H.sum(0))
    # plt.hist(x[:, 1], bins=30)

    C = np.cov(x, rowvar=0)
    s = np.sqrt(np.diag(C))
    print(C / np.outer(s, s))


def test_distorarrayparam():
    """DistOrArrayParams can be distributions or samples."""
    class Test(object):
        dp = dists.DistOrArrayParam(default=None, sample_shape=['*', '*'])

    inst = Test()
    inst.dp = dists.UniformHypersphere()
    assert isinstance(inst.dp, dists.UniformHypersphere)
    inst.dp = np.array([[1], [2], [3]])
    assert np.all(inst.dp == np.array([[1], [2], [3]]))
    with pytest.raises(ValueError):
        inst.dp = 'a'
    # Sample must have correct dims
    with pytest.raises(ValueError):
        inst.dp = np.array([1])


def test_distorarrayparam_sample_shape():
    """sample_shape dictates the shape of the sample that can be set."""
    class Test(object):
        dp = dists.DistOrArrayParam(default=None, sample_shape=['d1', 10])
        d1 = 4

    inst = Test()
    # Distributions are still cool
    inst.dp = dists.UniformHypersphere()
    assert isinstance(inst.dp, dists.UniformHypersphere)
    # Must be shape (4, 10)
    inst.dp = np.ones((4, 10))
    assert np.all(inst.dp == np.ones((4, 10)))
    with pytest.raises(ValueError):
        inst.dp = np.ones((10, 4))
    assert np.all(inst.dp == np.ones((4, 10)))


def test_frozen():
    """Test attributes inherited from FrozenObject"""
    a = dists.Uniform(-0.3, 0.6)
    b = dists.Uniform(-0.3, 0.6)
    c = dists.Uniform(-0.2, 0.6)

    assert hash(a) == hash(a)
    assert hash(b) == hash(b)
    assert hash(c) == hash(c)

    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert hash(a) != hash(c)  # not guaranteed, but highly likely
    assert b != c
    assert hash(b) != hash(c)  # not guaranteed, but highly likely
