"""utility functions for constructing and using Gaussian Processes to interpolate tabular data
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

#-------------------------------------------------

### simple kernels

class WhiteNoiseKernel(object):
    """a simple white-noise kernel:
    cov[f(x1), f(x2)] = sigma**2 * delta(x1-x2)
    """

    def __init__(self, sigma):
        self.sigma = sigma

    def cov(self, x1, x2):
        return self.sigma**2 * np.all(x1 == x2)

class SquaredExponentialKernel(object):
    """a simple Squared-Exponential kernel:
    cov[f(x1), f(x2)] = sigma**2 * exp(-(x1-x2)**2/length**2)
    """

    def __init__(self, sigma, *lengths):
        self.sigma = sigma
        self.lengths = np.array(lengths)

    def cov(self, x1, x2):
        return self.sigma**2 * np.exp(-np.sum((x1-x2)**2/self.lengths**2))

#------------------------

class CombinedKernel(object):
    """an object that represent the sum of multiple kernels
    """

    def __init__(self, *kernels):
        self.kernels = kernels

    def cov(self, *args, **kwargs):
        ans = 0.0
        for kernel in self.kernels:
            ans += kernel.cov(*args, **kwargs)
        return ans

#-------------------------------------------------

### logic to compute the mean, cov at points on a grid conditioned on sample points given a kernel

def x2cov(x1, x2, kernel):
    """a helper function that build covariance matrices
    """
    n1 = len(x1)
    n2 = len(x2)
    cov = np.empty((n1, n2), dtype=float)
    for i1 in range(n1):
        for i2 in range(n2):
            cov[i1, i2] = kernel.cov(x1[i1], x2[i2])
    return cov

def condition(target_x, source_x, source_f, kernel, verbose=False):
    """compute the mean and covariance of the function at target_x given the observations of the function \
at source_f = f(source_x) using the kernel and a zero-mean prior prior.
Based on Eq 2.19 of Rasmussen & Williams (2006) : http://gaussianprocess.org/gpml/chapters/RW.pdf
    """
    # compute the relevant blocks of the joint covariance matrix
    if verbose:
        print('constructing %d x %d target-target covariance matrix'%(len(target_x), len(target_x)))
    cov_tar_tar = x2cov(target_x, target_x, kernel)

    if verbose:
        print('constructing %d x %d target-source covariance matrix'%(len(target_x), len(source_x)))
    cov_tar_src = x2cov(target_x, source_x, kernel)

    if verbose:
        print('constructing %d x %d source-source covariance matrix'%(len(source_x), len(source_x)))
    cov_src_src = x2cov(source_x, source_x, kernel)

    # invert this covariance only once
    if verbose:
        print('inverting source-source covariance matrix')
    inv_cov_src_src = np.linalg.inv(cov_src_src)

    # compute the mean
    if verbose:
        print('computing conditioned mean')
    mean = cov_tar_src @ inv_cov_src_src @ source_f

    # compute the covariance
    if verbose:
        print('computing conditioned covariance')
    cov = cov_tar_tar - cov_tar_src @ inv_cov_src_src @ np.transpose(cov_tar_src)

    # return
    return mean, cov

#-------------------------------------------------

### optimization to find the best parameters for the kernel

def log_likelihood(source_x, source_f, kernel):
    """compute the marginal likelihood of observing source_f = f(source_x) given kernel and zero-mean process
    """
    cov_src_src = kernel.cov(source_x, source_x)
    s, logdet = np.linalg.slogdet(cov_src_src)
    assert s > 0, 'covariance is not positive definite!'

    # compute the log-likelihood
    return -0.5 * source_f @ np.linalg.inv(cov_src_src) @ source_f \
        - 0.5*logdet - 0.5*len(source_f)*np.log(2*np.pi)
