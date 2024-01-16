"""utility functions for constructing and using Gaussian Processes to interpolate tabular data
"""
__author__ = "Reed Essick (reed.essick@gmail.com), Ziyuan Zhang (ziyuan.z@wustl.edu)"

#-------------------------------------------------

import time

import warnings

import numpy as np

try:
    from scipy.special import gamma as _gamma
    from scipy.special import kv as _bessel_k
except:
    _gamma = _bessel_k = None

try:
    from scipy.optimize import minimize as _minimize
except:
    _minimize = None

try:
    import emcee as _emcee
except ImportError:
    _emcee = None

#-------------------------------------------------

### classes to represent simple kernels

class Kernel(object):
    """a parent class that defines the API for all kernel objects
    """
    _params = ()

    def __init__(self, *params):
        assert len(params) == len(self._params), 'must specify all parameters!\n\tparams=%s' % self._params
        self.params = np.array(params, dtype=float)

    @property
    def params_dict(self):
        return dict(zip(self._params, self.params))

    #---

    def __str__(self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join('%s=%.3e'%item for item in self.params_dict.items()))

    def __repr__(self):
        return self.__str__()

    #---

    def update(self, **params):
        """update the internal parameters that describe this kernel
        """
        for key, val in params.items():
            try:
                self.params[self._params.index(key)] = val
            except ValueError:
                warnings.warn('Warning! cannot update %s in object type %s' % (key, self.__class__.__name__))

    #---

    def cov(self, x1, x2):
        """compute the covariance between vectors x1 and x2 given the internal parameters of this kernel. \
This will return a matrix with shape (len(x1), len(x2))
        """
        raise NotImplementedError('this should be overwritten by child classes')

#------------------------

class NDKernel(Kernel):
    """a class that supports the kernels of variable dimension that require different numbers of parameters \
depending on the dimensionality.
    """

    def __init__(self, *lengths):
        self._params = ()
        self.params = ()
        self._parse_lengths(*lengths)
        Kernel.__init__(self, *self._params) # will cast to a standard type

    def _parse_lengths(self, *lengths):
        """update self._params and self.params to account for the correct dimensionality of the kernel
        """
        assert len(lengths), 'must specify at least one length'
        self._num_dim = len(lengths)
        self._params = self._params + tuple('length%d'%ind for ind in range(self._num_dim))
        self.params = self.params + tuple(lengths)

    @property
    def num_dim(self):
        return self._num_dim

#-------------------------------------------------

class WhiteNoiseKernel(Kernel):
    """a simple white-noise kernel:
    cov[f(x1), f(x2)] = sigma**2 * delta(x1-x2)
    """
    _params = ('sigma',)

    def cov(self, x1, x2):
        """expect x1, x2 to each have the shape : (Nsamp, Ndim)
        """
        return self.sigma**2 * np.all(x1 == x2, axis=1)

#------------------------

class MaternKernel(NDKernel):
    """a Matern covariance kernel: https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function
    the dimensionality of the kernel is set by the number of "lengths" specified upon instantiation
    """
    def __init__(self, order, sigma, *lengths):

        # check that we could import required functions
        if _gamma is None:
            raise ImportError('could not import scipy.special.gamma')
        if _bessel_k is None:
            raise ImportError('could not import scipy.special.kv')

        # now set up params
        self._params = ('order', 'sigma')
        self.params = (order, sigma)
        self._parse_lengths(*lengths)
        Kernel.__init__(self, *self.params)

    #---

    def cov(self, x1, x2):
        """exect x1, x2 to each have the shape : (Nsamp, Ndim)
        """
        o = self.params[0] # order
        s = self.params[1] # sigma
        lengths = self.params[2:]
        diff = (2*o)**0.5 * np.sum((x1-x2)**2/lengths**2, axis=1)**0.5
        return s**2 * (2**(1-o) / _gamma(o)) * diff**o * _bessel_k(o, diff)

#------------------------

class SquaredExponentialKernel(MaternKernel):
    """a simple Squared-Exponential kernel (the limit of Matern as order -> infty):
    cov[f(x1), f(x2)] = sigma**2 * exp(-(x1-x2)**2/length**2)
    """
    _params = ('sigma', 'lengths')
    
    def __init__(self, sigma, *lengths):
        self._params = ('sigma',)
        self.params = (sigma,)
        self._parse_lengths(*lengths)
        Kernel.__init__(self, *self.params)

    #---

    def cov(self, x1, x2):
        """expect x1, x2 to each have the shape : (Nsamp, Ndim)
        """
        sigma = self.params[0]
        lengths = self.params[1:]
        return sigma**2 * np.exp(-np.sum((x1-x2)**2/lengths**2, axis=1))

#------------------------

### a class to represent a mixture over kernels

class CombinedKernel(object):
    """an object that represent the sum of multiple kernels
    """

    def __init__(self, *kernels):

        # check that we have a reasonable number of kernels
        assert len(kernels) >= 2, 'must supply at least 2 kernels'
        self._num_kernels = len(kernels)

        # iterate over kernels, sanity checking and creating map for _params between kernels and their parameter names
        self._num_dim = None
        self._params = ()

        for ind, kernel in enumerate(kernels):

            # check that dimensionality agrees between all kernels
            if isinstance(kernel, NDKernel):
                if num_dim is None:
                    self._num_dim = kernel.num_dim
                else:
                    assert self._num_dim == kernel.num_dim, 'conflict in dimensionality of kernels!'

            # add parameters to the tuple
            self._params = self._params + tuple(self.combinedkernel_name(name, ind) for name in kernel._params)

        self.kernels = tuple(kernels) # make a new object so we don't mess up with cross-references

    #---

    @staticmethod
    def _combinedkernel_name(name, index):
        return '%s_%s' % (name, index)

    @staticmethod
    def _kernel_name(name):
        name = name.split('_')
        return '_'.join(name[:-1]), int(name[-1])
        
    #---

    def __str__(self):
        ans = self.__class__.__name__
        for ind, kernel in enumerate(self.kernels):
            ans += '\n\t%d\t%s' % (ind, str(kernel))

    def __repr__(self):
        return self.__str__()

    #---

    def update(self, **params):
        """update each kernel in turn
        """
        # map the parameters into smaller dictionaries for separate kernels
        ans = defaultdict(dict)
        for key, val in params.items():
            name, ind = self._kernel_name(key)
            ans[ind][name] = val

        # now iterate and update each kernel in turn
        for ind, params in ans.items():
            self.kernels[ind].update(**params)

    #---

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

    # utilities for representing the mean of the conditioned process efficiently

    def compress(self, source_x, source_f, verbose=False, Verbose=False):
        """compress the GP mean prediction into a single array of the same length as the training set
    return inv(Cov(source_x, source_x)) @ source_f
        """

        # construct covariane matrix
        if verbose:
            print('constructing %d x %d source-source covariance matrix'%(len(source_x), len(source_x)))
            t0 = time.time()
        cov_src_src = self._x2cov(source_x, source_x, self.kernel, verbose=Verbose)
        if verbose:
            print('    time : %.6f sec' % (time.time()-t0))

        # invert this covariance only once
        if verbose:
            print('inverting source-source covariance matrix')
            t0 = time.time()
        inv_cov_src_src = np.linalg.inv(cov_src_src)
        if verbose:
            print('    time : %.6f sec' % (time.time()-t0))

        # compute the contraction that can be used to compute the mean
        if verbose:
            print('compressing observations')
            t0 = time.time()
        compressed = inv_cov_src_src @ source_f
        if verbose:
            print('    time : %.6f sec' % (time.time()-t0))

        # return
        return compressed

    #---

    def predict(self, target_x, source_x, compressed, verbose=False, Verbose=False):
        """used the compressed representation of the training data to predict the mean at target_x
        """

        # construct covariane matrix
        if verbose:
            print('constructing %d x %d target-source covariance matrix'%(len(target_x), len(source_x)))
            t0 = time.time()
        cov_tar_src = self._x2cov(target_x, source_x, self.kernel, verbose=Verbose)
        if verbose:
            print('    time : %.6f sec' % (time.time()-t0))

        # compute the mean
        if verbose:
            print('computing conditioned mean')
            t0 = time.time()
        mean = cov_tar_src @ compressed
        if verbose:
            print('    time : %.6f sec' % (time.time()-t0))

        # return
        return mean

    #--------------------

    # general utilities for full GP regression

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

        # return
        return mean, cov

    #---

    @staticmethod
    def _x2cov(x1, x2, kernel, verbose=False):
        """a helper function that build covariance matrices
        """

        # check dimensionality of the data
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

        # return
        return cov

    #--------------------

    # utilities for drawing samples from the GP model

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
        scales = np.diag(cov)**0.5 # re-scale variables to make drawing samples more numerically stable
        zeros = np.zeros_like(mean, dtype=float)
        rands = np.random.multivariate_normal(zeros, cov/np.outer(scales, scales), size=size) * scales

        # add fluctuations to the mean function and return
        return mean + rands

    #--------------------

    # utilities for determining good hyperparameters for the model
    # these are based on the marginal likelihood for the observed data conditioned on the kernel parameters

    def loglikelihood(self, source_x, source_f):
        """compute the marginal likelihood of observing source_f = f(source_x) given kernel and zero-mean process
        """
        cov_src_src = self._x2cov(source_x, source_x, self.kernel, verbose=verbose)
        s, logdet = np.linalg.slogdet(cov_src_src)
        assert s > 0, 'covariance is not positive definite!'

        # compute the log-likelihood
        return -0.5 * source_f @ np.linalg.inv(cov_src_src) @ source_f - 0.5*logdet - 0.5*len(source_f)*np.log(2*np.pi)

    #---

    def optimize_kernel(self, source_x, source_f, verbose=False): #, bound_list):
        """
        Find the set of parameters for the kernel that maximize loglikelihood(source_x, source_f) via scipy.optimize.minimize
        """
        if _minimize is None:
            raise ImportError('could not import scipy.optimize.minimize')

        # Minimize the negative loglikelihood (maximize loglikelihood)

        ## define target function that we will minimize
        def target(**params):
            for key, val in params.items(): # check to make sure parameters are reasonable
                if val < 0:
                    return np.infty # return a big number so we avoid this region
            self.update(**params)
            return - self.loglikelihood(source_x, source_f)

        # FIXME! check to see whether "bound_list" is actually needed...
#        bounds = [bound_list[key] for key in initial_params]

        ## run the minimizer
        result = _minimize(
            target,
            self.kernel.params_dict,
#            bounds=(bounds),
            method='TNC',
        )

        # update the kernel to match the optimal parameters
        self.kernel.update(**dict(zip(self.kernel._params, result.x)))

        # return
        return self.kernel.params_dict

    #---

    def sample_kernel(self, *args, **kwargs):
        raise NotImplementedError('''


    def sample_kernel(self, source_x, source_f, init_para_dict, burn_in=100, nsteps=100, nwalkers=None):
        """
        Sample the kernel parameters from a distribution defined by loglikelihood(source_x, source_f)
        
        :param init_para: Dictionary of initial guess for parameters, {"sigma": sigma, "length1": length1, etc.}
        :param nwalkers: Number of walkers for the MCMC sampler
        :param burn_in: Number of burn-in steps for the MCMC sampler
        :param nsteps: Number of steps to run the MCMC sampler
        
        :return: the MCMC sampler object
        """
        if _emcee is None:
            raise ImportError('could not import emcee')

        # raise NotImplementedError('should be overwritten by child classes')

        # initial parameters
        init_para = list(init_para_dict.values())

        n_dim = len(init_para_dict)
        if nwalkers is None:
            nwalkers = 2*n_dim
            # print("check point 1")
            # print("nwalkers = ", nwalkers)

        # Initialize walkers
        bounds = [(param - param / 8, param + param / 8) for param in init_para]

        walkers = np.array([np.random.uniform(low, high, nwalkers) for low, high in bounds]).T
        # print("walkers: ")
        # print(walkers)

        # give sampler negative_log_posterior
        # sampler = emcee.EnsembleSampler(nwalkers, n_dim, self.negative_log_posterior)
        sampler = emcee.EnsembleSampler(nwalkers, n_dim,
                                        lambda params: self.negative_log_likelihood(source_x, source_f,
                                                                      dict(zip(init_para_dict.keys(), params)))
                                        )

        # burn-in steps
        print('running MCMC burn-in with {:d} steps'.format(burn_in))
        start = time.time()
        state = sampler.run_mcmc(walkers, burn_in)
        sampler.reset()
        end = time.time()
        print('    time : %.6f sec' % (end - start))

        # Reset and run the main MCMC sampling
        # at each step, calculate the log posterior at 
        # the current position and the new position
        # update or not using Metropolis-Hastings
        print('running MCMC sampling with {:d} steps'.format(nsteps))
        start = time.time()
        sampler.run_mcmc(state, nsteps)
        end = time.time()
        print('    time : %.6f sec' % (end - start))

        # Extract samples
        # samples = sampler.get_chain(flat=True)

        print("Mean acceptance fraction: {0:.3f}".format(
                np.mean(sampler.acceptance_fraction))
        )

        # print("Mean autocorrelation time: {0:.3f} steps".format(
        #         np.mean(sampler.get_autocorr_time()))
        # )

        # self.autocorr_time = sampler.get_autocorr_time()

        # Optionally, calculate and append log likelihood values
        # print('calculating likelihood for all samples in the chain')
        # start = time.time()
        # LLH_value = np.apply_along_axis(lambda row: self.negative_log_likelihood(row), axis=1, arr=samples)
        # end = time.time()
        # print('    time : %.6f sec' % (end - start))
        # samples_with_LLH = np.column_stack((samples, LLH_value))

        # randomly draw some? or have this part in the test?

        # return #samples_with_LLH
        # return samples
        return sampler
''')

#------------------------

class NearestNeighborInterpolator(Interpolator):
    """implements a NearestNeighbor Gaussian Process, which induces a sparse covariance matrix and allows for \
matrix inversion in linear time.
This is based on:
    Abhirup Datta, Sudipto Banerjee, Andrew O. Finley & Alan E. Gelfand (2016) Hierarchical Nearest-Neighbor Gaussian Process Models for Large Geostatistical Datasets, Journal of the American Statistical Association, 111:514, 800-812, DOI: 10.1080/01621459.2015.1044091
    """
    pass # FIXME! implement NearestNeighbor logic
