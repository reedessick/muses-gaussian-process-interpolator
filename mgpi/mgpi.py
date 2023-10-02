"""utility functions for constructing and using Gaussian Processes to interpolate tabular data
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import time

import numpy as np
from scipy.special import gamma
from scipy.special import kv as bessel_k

#-------------------------------------------------

### classes to represent simple kernels

class Kernel(object):
    """a parent class that defines the API for all kernel objects
    """

    def __init__(self, *params):
        self.params = params

    def update(self, **params):
        """update the internal parameters that describe this kernel
        """
        raise NotImplementedError('this should be implemented here and inherited by all child classes')

    def cov(self, x1, x2):
        """compute the covariance between vectors x1 and x2 given the internal parameters of this kernel. \
This will return a matrix with shape (len(x1), len(x2))
        """
        raise NotImplementedError('this should be overwritten by child classes')

#------------------------

class WhiteNoiseKernel(Kernel):
    """a simple white-noise kernel:
    cov[f(x1), f(x2)] = sigma**2 * delta(x1-x2)
    """

    def __init__(self, sigma):
        self.sigma = sigma

    def cov(self, x1, x2):
        """expect x1, x2 to each have the shape : (Nsamp, Ndim)
        """
        return self.sigma**2 * np.all(x1 == x2, axis=1)

class SquaredExponentialKernel(Kernel):
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

class MaternKernel(Kernel):
    """a Matern covariance kernel: https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function
    """

    def __init__(self, sigma, order, *lengths):
        self.sigma = sigma
        self.order = order
        self.lengths = np.array(lengths)

    def cov(self, x1, x2):
        """exect x1, x2 to each have the shape : (Nsamp, Ndim)
        """
        o = self.order
        diff = (2*o)**0.5 * np.sum((x1-x2)**2/self.length**2, axis=1)**0.5
        return self.sigma**2 * (2**(1-o) / gamma(o)) * diff**o * bessel_k(o, diff)

#------------------------

### a class to represent a mixture over kernels

class CombinedKernel(object):
    """an object that represent the sum of multiple kernels
    """

    def __init__(self, *kernels):
        self.kernels = kernels

    def update(self, **params):
        """update each kernel in turn
        """
        for k in self.kernels:
            k.update(**params)

    def cov(self, *args, **kwargs):
        """iterate over contained kernels and sum the corresponding covariances
        """
        ans = 0.0
        for kernel in self.kernels:
            ans += kernel.cov(*args, **kwargs)
        return ans

#-------------------------------------------------

### classes to perform interpolation based on kernels

#-------------------------------------------------

class Interpolator(object):
    """implements the most general Gaussian Process regression without assuming anything special \
about the structure of the covariance matrix or mean function
    """

    def __init__(self, kernel):
        self.kernel = kernel

    #--------------------

    def condition(self, target_x, source_x, source_f, verbose=False, Verbose=False):
        """compute the mean and covariance of the function at target_x given the observations of the function \
at source_f = f(source_x) using the kernel and a zero-mean prior prior.
Based on Eq 2.19 of Rasmussen & Williams (2006) : http://gaussianprocess.org/gpml/chapters/RW.pdf
        """
        verbose |= Verbose

        # compute the relevant blocks of the joint covariance matrix
        if verbose:
            print('constructing %d x %d target-target covariance matrix'%(len(target_x), len(target_x)))
            t0 = time.time()
        cov_tar_tar = self._x2cov(target_x, target_x, self.kernel, verbose=Verbose)
        if verbose:
            print('    time : %.6f sec' % (time.time()-t0))

        #---

        if verbose:
            print('constructing %d x %d target-source covariance matrix'%(len(target_x), len(source_x)))
            t0 = time.time()
        cov_tar_src = self._x2cov(target_x, source_x, self.kernel, verbose=Verbose)
        if verbose:
            print('    time : %.6f sec' % (time.time()-t0))

        #---

        if verbose:
            print('constructing %d x %d source-source covariance matrix'%(len(source_x), len(source_x)))
            t0 = time.time()
        cov_src_src = self._x2cov(source_x, source_x, self.kernel, verbose=Verbose)
        if verbose:
            print('    time : %.6f sec' % (time.time()-t0))

        #---

        # invert this covariance only once
        if verbose:
            print('inverting source-source covariance matrix')
            t0 = time.time()
        inv_cov_src_src = np.linalg.inv(cov_src_src)
        if verbose:
            print('    time : %.6f sec' % (time.time()-t0))

        #---

        # compute the mean
        if verbose:
            print('computing conditioned mean')
            t0 = time.time()
        mean = cov_tar_src @ inv_cov_src_src @ source_f
        if verbose:
            print('    time : %.6f sec' % (time.time()-t0))

        #---

        # compute the covariance
        if verbose:
            print('computing conditioned covariance')
            t0 = time.time()
        cov = cov_tar_tar - cov_tar_src @ inv_cov_src_src @ np.transpose(cov_tar_src)
        if verbose:
            print('    time : %.6f sec' % (time.time()-t0))

        #---

        # return
        return mean, cov

    #---

    @staticmethod
    def _x2cov(x1, x2, kernel, verbose=False):
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

        if verbose:
            print('constructing X1')
            t0 = time.time()

        X1 = np.empty((n1*n2, ndim), dtype=float)
        for ind in range(n1): # FIXME can I avoid this loop? it's not a big cost computationally...
            X1[ind*n2:(ind+1)*n2,:] = x1[ind,:]

        if verbose:
            print('    time : %.6f sec' % (time.time()-t0))

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

        if verbose:
            print('constructing X2')
            t0 = time.time()

        X2 = np.outer(np.ones(n1), x2).reshape((nsmp, ndim))

        if verbose:
            print('    time : %.6f sec' % (time.time()-t0))

        #---

        # finally, compute the covariance

        if verbose:
            print('computing covariance matrix')
            t0 = time.time()

        cov = kernel.cov(X1, X2)     # returns shape : Nsmp = n1*n2
        cov = cov.reshape((n1, n2))  # reshape to the desired input shape

        if verbose:
            print('    time : %.6f sec' % (time.time()-t0))

        #---

        return cov

    #--------------------

    def rvs(self, target_x, source_x, source_f, size=1):
        """return realizations of the process for f at target_x conditioned on the values of the function \
source_f at source_x
        """
        return self._rvs_from_conditioned(*self.condition(target_x, source_x, source_f), size=size)

    #---

    def _rvs_from_conditioned(mean, cov, size=1):
        """a helper function that draws realizations from a multivariate distribution defined by \
a mean function and a covariance matrix
        """
        # draw the random fluctuations around the mean function
        scales = np.diag(cov)**0.5
        zeros = np.zeros_like(mean, dtype=float)
        rands = np.random.multivariate_normal(zeros, cov/np.outer(scales, scales), size=size) * scales

        # add fluctuations to the mean function and return
        return mean + rands

    #--------------------

    def loglikelihood(self, source_x, source_f):
        """compute the marginal likelihood of observing source_f = f(source_x) given kernel and zero-mean process
        """
        cov_src_src = self.kernel.cov(source_x, source_x)
        s, logdet = np.linalg.slogdet(cov_src_src)
        assert s > 0, 'covariance is not positive definite!'

        # compute the log-likelihood
        return -0.5 * source_f @ np.linalg.inv(cov_src_src) @ source_f \
            - 0.5*logdet - 0.5*len(source_f)*np.log(2*np.pi)

    #---

    def optimize_kernel(self, source_x, source_f):
        """find the set of parameters for the kernel that maximize loglikelihood(source_x, source_f)
        """
        raise NotImplementedError('should be overwritten by child classes')

    #---

    def sample_kernel(self, source_x, source_f):
        """sample the kernel parameters from a distribution defined by loglikelihood(source_x, source_f)
        """
        raise NotImplementedError('should be overwritten by child classes')

#------------------------

class NearestNeighborInterpolator(Interpolator):
    """implements a NearestNeighbor Gaussian Process, which induces a sparse covariance matrix and allows for \
matrix inversion in linear time
    """
