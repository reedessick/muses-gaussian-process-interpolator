"""utility functions for constructing and using Gaussian Processes to interpolate tabular data
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import time

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
        """expect x1, x2 to each have the shape : (Nsamp, Ndim)
        """
        return self.sigma**2 * np.all(x1 == x2, axis=1)

class SquaredExponentialKernel(object):
    """a simple Squared-Exponential kernel:
    cov[f(x1), f(x2)] = sigma**2 * exp(-(x1-x2)**2/length**2)
    """

    def __init__(self, sigma, *lengths):
        self.sigma = sigma
        self.lengths = np.array(lengths)

    def cov(self, x1, x2):
        """expect x1, x2 to each have the shape : (Nsamp, Ndim)
        """
        return self.sigma**2 * np.exp(-np.sum((x1-x2)**2/self.lengths**2, axis=1))

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

    nsmp = n1*n2
    ndim = len(x1[0])
    assert len(x2[0]) == ndim, 'conflicting number of dimensions for x1 and x2'

    #---

    # do the following conversion from
    #   x1 = [[b00, b01, ..., b0N],
    #         [b10, b11, ..., b1N],
    #           ...
    #         [bY0, bN1, ..., bYN]]
    # to
    #   X1 = [[b00, b01, ..., b0N], # this repeats X=len(x2) times, and there are Y=len(x1) such blocks
    #         [b00, b01, ..., b0N],
    #           ...
    #         [b00, b01, ..., b0N],
    #         [b10, b11, ..., b1N], # this repeats X=len(x2) times
    #         [b10, b11, ..., b1N],
    #           ...
    #         [b10, b11, ..., b1N],
    #         ...
    #         [bY0, aY1, ..., bYN], # this repeats X=len(x2) times
    #         [bY0, aY1, ..., bYN],
    #           ...
    #         [bY0, aY1, ..., bYN],

    X1 = np.empty((n1*n2, ndim), dtype=float)
    for ind in range(n1): # FIXME can I avoid this loop?
        X1[ind*n2:(ind+1)*n2,:] = x1[ind,:]

    #---

    # do the following conversion from
    #   x2 = [[a00, a01, ..., a0N],
    #         [a10, a11, ..., a1N],
    #           ...
    #         [aX0, aX1, ..., aXN]]
    # to
    #   X2 = [[a00, a01, ..., a0N],
    #         [a10, a11, ..., a1N],
    #           ...
    #         [aX0, aX1, ..., aXN], # there are X=len(x2) rows up to this point
    #         [a00, a01, ..., a0N], # this repeats Y=len(x1) times
    #         [a10, a11, ..., a1N],
    #           ...
    #         [aX0, aX1, ..., aXN],
    #           ...
    #         [a00, a01, ..., a0N],
    #         [a10, a11, ..., a1N],
    #           ...
    #         [aX0, aX1, ..., aXN]]

    X2 = np.outer(np.ones(n1), x2).reshape((nsmp, ndim))

    #---

    # finally, compute the covariance
    cov = kernel.cov(X1, X2)     # returns shape : Nsmp = n1*n2
    cov = cov.reshape((n1, n2))  # reshape to the desired input shape

    #---

    return cov

def condition(target_x, source_x, source_f, kernel, verbose=False):
    """compute the mean and covariance of the function at target_x given the observations of the function \
at source_f = f(source_x) using the kernel and a zero-mean prior prior.
Based on Eq 2.19 of Rasmussen & Williams (2006) : http://gaussianprocess.org/gpml/chapters/RW.pdf
    """
    # compute the relevant blocks of the joint covariance matrix
    if verbose:
        print('constructing %d x %d target-target covariance matrix'%(len(target_x), len(target_x)))
        t0 = time.time()

    cov_tar_tar = x2cov(target_x, target_x, kernel)

    if verbose:
        print('    time : %.6f sec' % (time.time()-t0))

    if verbose:
        print('constructing %d x %d target-source covariance matrix'%(len(target_x), len(source_x)))
        t0 = time.time()

    cov_tar_src = x2cov(target_x, source_x, kernel)

    if verbose:
        print('    time : %.6f sec' % (time.time()-t0))

    if verbose:
        print('constructing %d x %d source-source covariance matrix'%(len(source_x), len(source_x)))
        t0 = time.time()

    cov_src_src = x2cov(source_x, source_x, kernel)

    if verbose:
        print('    time : %.6f sec' % (time.time()-t0))

    # invert this covariance only once
    if verbose:
        print('inverting source-source covariance matrix')
        t0 = time.time()

    inv_cov_src_src = np.linalg.inv(cov_src_src)

    if verbose:
        print('    time : %.6f sec' % (time.time()-t0)) 

    # compute the mean
    if verbose:
        print('computing conditioned mean')
        t0 = time.time()

    mean = cov_tar_src @ inv_cov_src_src @ source_f

    if verbose:
        print('    time : %.6f sec' % (time.time()-t0))

    # compute the covariance
    if verbose:
        print('computing conditioned covariance')
        t0 = time.time()

    cov = cov_tar_tar - cov_tar_src @ inv_cov_src_src @ np.transpose(cov_tar_src)

    if verbose:
        print('    time : %.6f sec' % (time.time()-t0))

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
