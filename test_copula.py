import numpy as np
import scipy.stats

# R = np.array([[ 1.0000, -0.2721,  0.0042, -0.5636, -0.1307],
#               [-0.2721,  1.0000, -0.1357,  0.4244, -0.4509],
#               [ 0.0042, -0.1357,  1.0000,  0.4765, -0.0464],
#               [-0.5636,  0.4244,  0.4765,  1.0000, -0.3097],
#               [-0.1307, -0.4509, -0.0464, -0.3097,  1.0000]])

R = np.array([[1., 0.5], [0.5, 1.]])

assert np.array_equal(R, R.T)


n = 1000000
U = scipy.stats.norm.cdf(scipy.stats.multivariate_normal.rvs(cov=R, size=n))

C = np.cov(U, rowvar=0)
print(C)
s = np.sqrt(np.diag(C))
print(C / np.outer(s, s))
# print(C / C[0, 0])

U = scipy.stats.norm.cdf(scipy.stats.multivariate_normal.rvs(cov=C, size=n))
C = np.cov(U, rowvar=0)
print(C)


# print(unis)




# print sample
# r = np.linalg.cholesky(R)
# print(r)
