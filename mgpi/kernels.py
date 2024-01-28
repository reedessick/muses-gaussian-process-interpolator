"""kernels for Gaussian Process models
"""
__author__ = "Reed Essick (reed.essick@gmail.com), Ziyuan Zhang (ziyuan.z@wustl.edu)"

#-------------------------------------------------

import time
import warnings

from collections import defaultdict

import numpy as np

try:
    from scipy.special import gamma as _gamma
    from scipy.special import kv as _bessel_k
except:
    _gamma = _bessel_k = None

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

    @property
    def params_array(self):
        ans = np.empty(1, dtype=[(_,float) for _ in self._params])
        for key, val in zip(self._params, self.params):
            ans[key] = val
        return ans

    #---

    def __str__(self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join('%s=%.6e'%item for item in self.params_dict.items()))

    def __repr__(self):
        return self.__str__()

    #---

    def __add__(self, other):
        """return a CombinedKernel
        """
        kernels = (self.kernels if isinstance(self, CombinedKernel) else [self]) \
            + (other.kernels if isinstance(other, CombinedKernel) else [other])
        return CombinedKernel(*kernels)

    #---

    def update(self, *args, **params):
        """update the internal parameters that describe this kernel
        """
        num_args = len(args)

        if num_args:
            if params:
                raise ValueError('cannot update with both args and params at the same time!')

            elif num_args == len(self._params): # assume we just passed a vector
                self.params[:] = args

            elif (num_args == 1) and isinstance(args[0], dict): # handle a dictionary
                self.update(**args[0])

            else:
                raise ValueError('could not interpret args=%s'%args)
        else:
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
        return self.params[0]**2 * np.all(x1 == x2, axis=1)

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

class CombinedKernel(Kernel):
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
                if self._num_dim is None:
                    self._num_dim = kernel.num_dim
                else:
                    assert self._num_dim == kernel.num_dim, 'conflict in dimensionality of kernels!'

            # add parameters to the tuple
            self._params = self._params + tuple(self._combinedkernel_name(name, ind) for name in kernel._params)

        self.kernels = tuple(kernels) # make a new object so we don't mess up with cross-references

    #---

    @property
    def params(self):
        return np.concatenate([kernel.params for kernel in self.kernels])

    @staticmethod
    def _combinedkernel_name(name, index):
        return '%s_%s' % (name, index)

    @staticmethod
    def _kernel_name(name):
        name = name.split('_')
        try:
            ind = int(name[-1])
        except ValueError:
            raise RuntimeError('cannot map "%s" to parameter name and kernel index!' % ('_'.join(name)))

        return '_'.join(name[:-1]), ind
        
    #---

    def __str__(self):
        ans = self.__class__.__name__
        for ind, kernel in enumerate(self.kernels):
            ans += '\n    kernel %-2d : %s' % (ind, str(kernel))
        return ans

    def __repr__(self):
        return self.__str__()

    #---

    def update(self, *args, **params):
        """update each kernel in turn
        """
        num_args = len(args)

        if num_args:
            if params:
                raise ValueError('cannot update with both args and params at the same time!')

            elif num_args == len(self._params): # assume we just passed a vector
                start = 0
                for kernel in self.kernels:
                    stop = start + len(kernel._params)
                    kernel.update(*args[start:stop])
                    start = stop

            elif (num_args == 1) and isinstance(args[0], dict): # handle a dictionary
                self.update(**args[0])

            else:
                raise ValueError('could not interpret args=%s'%args)

        else:
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
