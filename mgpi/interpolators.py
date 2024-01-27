"""object-oriented Gaussian Processes interpolators for tabular data
"""
__author__ = "Reed Essick (reed.essick@gmail.com), Ziyuan Zhang (ziyuan.z@wustl.edu)"

#-------------------------------------------------

import time

import numpy as np

try:
    from scipy.optimize import minimize as _minimize
except:
    _minimize = None

try:
    import emcee as _emcee
except ImportError:
    _emcee = None

#-------------------------------------------------

# default method for optimization

DEFAULT_METHOD = 'TNC' # used within scipy.optimize.minimize

#------------------------

# default parameters for MCMC sampling

DEFAULT_TEMPERATURE = 1.0 # used to temper the likelihood within logprob
DEFAULT_NUM_BURNIN = 100
DEFAULT_NUM_SAMPLES = 1000
DEFAULT_NUM_WALKERS = None # will set num_walkers based on the dimensionality of the sampling problem

#------------------------

# default parameters for nearest-neighbor interpolator logic

DEFAULT_NUM_NEIGHBORS = 10
DEFAULT_ORDER_BY_INDEX = None

#-------------------------------------------------

### classes to perform interpolation based on kernels

class Interpolator(object):
    """implements the most general Gaussian Process regression without assuming anything special \
about the structure of the covariance matrix or mean function
    """

    def __init__(self, kernel, nugget=None):
        self.kernel = kernel
        self.nugget = nugget

    def update(self, *args, **kwargs):
        """a convenience function for updating kernel parameters
        """
        return self.kernel.update(*args, **kwargs)

    def update_nugget(self, *args, **kwargs):
        """update the nugget's parameters
        """
        if self.nugget is None:
            raise RuntimeError('cannot update nugget=None')
        self.nugget.update(*args, **kwargs)

    #--------------------

    # utilities for representing the mean of the conditioned process efficiently

    def compress(self, source_x, source_f, verbose=False, Verbose=False):
        """compress the GP mean prediction into a single array of the same length as the training set
    return inv(Cov(source_x, source_x)) @ source_f
        """

        # construct covariance matrix
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

    def loglikelihood(self, source_x, source_f, verbose=False):
        """compute the marginal likelihood of observing source_f = f(source_x) given kernel and zero-mean process
        """
        cov_src_src = self._x2cov(source_x, source_x, self.kernel, verbose=verbose)
        s, logdet = np.linalg.slogdet(cov_src_src)
        assert s > 0, 'covariance is not positive definite!'

        # compute the log-likelihood
        return -0.5 * source_f @ np.linalg.inv(cov_src_src) @ source_f - 0.5*logdet - 0.5*len(source_f)*np.log(2*np.pi)

    #---

    def _construct_logprob(
            self,
            source_x,
            source_f,
            logprior=None,
            fixed=None,
            temperature=DEFAULT_TEMPERATURE,
            verbose=False,
            **kwargs # extra kwargs handed to self.loglikelihood within logprob
        ):
        """construct a target function suitable for optimize_kernel and sample_kernel
        """
        ## define target function that we will minimize
        if logprior is None: # set prior to be flat
            logprior = lambda x: 0.0

        _params = self.kernel._params
        if fixed is not None:
            self.update(**fixed) # set the parameters to their fixed values
            _params = [name for name in _params if name not in fixed] # only update the un-fixed params within logprob
        
        def logprob(params):
            # check parameters to make sure they are reasonable
            if any(params <= 0) or any(params != params):
                return -np.infty # avoid this region

            self.update(**dict(zip(_params, params)))

            # evaluate prior to make sure this point is allowed
            logp = logprior(params)
            if logp == -np.infty:
                return -np.infty # don't bother evaluating likelihood

            # evaulate likelihood
            logl = self.loglikelihood(source_x, source_f, **kwargs) / temperature # temper the likelihood

            # report and return
            if verbose:
                print('>>> %s\n  logl=%.6e\n  logp=%.6e' % (self.kernel, logl, logp))
            return logl + logp

        return logprob

    def _construct_initial_params(self, logprior=None, fixed=None, size=1, verbose=False):
        """generate initial locations for optimization and/or sampling algorithms
        """
        num_params = len(self.kernel.params)
        num_fixed = len(fixed) if fixed is not None else 0
        num_dim = num_params - num_fixed

        if verbose:
            print('initializing %d samples with num_dim = %d (%d params - %d fixed)' % \
                (size, num_dim, num_params, num_fixed))
            t0 = time.time()

        # scatter parameters in a unit ball around the initial guess
        state = np.empty((size, num_dim), dtype=float)
        n = 0 # the number of accepted points

        if verbose:
            trials = 0 # the number of tries

        _params = self.kernel.params
        if fixed is not None: # only generate samples for the params that are not fixed
            _params = [val for key, val in zip(self.kernel._params, self.kernel.params) if (key not in fixed)] 

        while n < size:
            if verbose:
                trials += 1

            # draw parameters
            params = _params * (1 + np.random.normal(size=num_dim))

            # sanity check them
            if np.any(params <= 0): # do not allow negative params
                continue
            if (logprior is not None) and (logprior(params) == -np.infty): # don't keep this sample; the prior disallows it
                continue

            # record this sample as it passes all sanity checks
            state[n] = params
            n += 1

        if verbose:
            print('    time : %.6f sec (%d/%d trials accepted)' % (time.time()-t0, n, trials))

        # return
        return state

    #--------------------

    def optimize_kernel(
            self,
            source_x,
            source_f,
            method=DEFAULT_METHOD,
            logprior=None,
            fixed=None,
            temperature=DEFAULT_TEMPERATURE,
            verbose=False,
            Verbose=False,
        ):
        """
        Find the set of parameters for the kernel that maximize loglikelihood(source_x, source_f) via scipy.optimize.minimize
        """
        if _minimize is None:
            raise ImportError('could not import scipy.optimize.minimize')

        # Minimize the negative loglikelihood (maximize loglikelihood)
        logprob = self._construct_logprob(
            source_x,
            source_f,
            logprior=logprior,
            fixed=fixed,
            temperature=temperature,
            verbose=Verbose,
        )

        # construct initial parameters
        intial_params = self._construct_initial_params(logprior=logprior, fixed=fixed, size=1, verbose=Verbose)[0]

        ## run the minimizer
        if verbose:
            print('extremizing loglikelihood')
            t0 = time.time()

        result = _minimize(
            lambda params: -logprob(params), # minimize the negative loglike --> maximize loglike
            initial_params,
            method=method,
        )

        if verbose:
            print('    time : %.6f sec' % (time.time()-t0))

        # update the kernel to match the optimal parameters
        self.update(*result.x)

        # return
        return self.kernel.params_dict

    #--------------------

    def _instantiate_sampler(
            self,
            source_x,
            source_f,
            logprior=None,
            fixed=None,
            temperature=DEFAULT_TEMPERATURE,
            num_walkers=DEFAULT_NUM_WALKERS,
            verbose=False,
            Verbose=False,
        ):

        if _emcee is None:
            raise ImportError('could not import emcee')

        verbose |= Verbose

        # check dimensionality of the sampling problem
        num_params = len(self.kernel.params)
        num_fixed = len(fixed) if fixed is not None else 0

        num_dim = num_params - num_fixed

        if num_walkers is None:
            num_walkers = 2*num_dim

        # instantiate sampler
        if verbose:
            print('initializing sampler\n    %d walkers\n    %d dimensions (%d params - %d fixed)\n    temperature=%.3e' % \
                (num_walkers, num_dim, num_params, num_fixed, temperature))
            t0 = time.time()

        ## define the target distribution (loglikelihood)
        logprob = self._construct_logprob(
            source_x,
            source_f,
            logprior=logprior,
            fixed=fixed,
            temperature=temperature,
            verbose=Verbose,
        )

        ## instantiate the sampler
        sampler = _emcee.EnsembleSampler(num_walkers, num_dim, logprob)

        if verbose:
            print('    time : %.6f sec' % (time.time()-t0))

        # return
        return sampler, (num_dim, num_walkers)

    #---

    def sample_kernel(
            self,
            source_x,
            source_f,
            logprior=None,
            fixed=None,
            temperature=DEFAULT_TEMPERATURE,
            num_burnin=DEFAULT_NUM_BURNIN,
            num_samples=DEFAULT_NUM_SAMPLES,
            num_walkers=DEFAULT_NUM_WALKERS,
            verbose=False,
            Verbose=False,
        ):
        """
        Sample the kernel parameters from a distribution defined by loglikelihood(source_x, source_f)
        """
        verbose |= Verbose

        #---

        # set up the sampler
        sampler, (num_dim, num_walkers) = self._instantiate_sampler(
            source_x,
            source_f,
            logprior=logprior,
            fixed=fixed,
            temperature=temperature,
            num_walkers=num_walkers,
            verbose=verbose,
        )

        #---

        # picking initial positions for walkers
        state = self._construct_initial_params(logprior=logprior, fixed=fixed, size=num_walkers, verbose=verbose)

        #---

        # running burn-in
        if verbose:
            print('running burn-in with %d steps' % num_burnin)
            t0 = time.time()

        state = sampler.run_mcmc(state, num_burnin, progress=Verbose)
        sampler.reset() # remove burn-in samples from internal bookkeeping

        if verbose:
            print('    time : %.6f sec' % (time.time()-t0))

        #---

        # generate production samples

        if verbose:
            print('drawing %d samples' % num_samples)
            t0 = time.time()

        sampler.run_mcmc(state, num_samples, progress=Verbose)

        if verbose:
            print('    time : %.6f sec' % (time.time()-t0))

        # return
        samples = sampler.get_chain() # array with shape: (num_samples, num_walkers, num_dim)
        logprob = sampler.get_log_prob() # array with shape: (num_samples, num_walkers)

        return samples, logprob, sampler

#------------------------

class NearestNeighborInterpolator(Interpolator):
    """implements a NearestNeighbor Gaussian Process, which induces a sparse covariance matrix and allows for \
matrix inversion in linear time.
This is based on:
    Abhirup Datta, Sudipto Banerjee, Andrew O. Finley & Alan E. Gelfand (2016)
    Hierarchical Nearest-Neighbor Gaussian Process Models for Large Geostatistical Datasets,
    Journal of the American Statistical Association, 111:514, 800-812, 
    DOI: 10.1080/01621459.2015.1044091
    """

    def __init__(self, kernel, num_neighbors=DEFAULT_NUM_NEIGHBORS, order_by_index=DEFAULT_ORDER_BY_INDEX):
        self.num_neighbors = num_neighbors   # the number of neighbors retained in the algorithm
        self.order_by_index = order_by_index # order samples by the values in this index
        Interpolator.__init__(self, kernel)

    #---

    # methods pecular to the NNGP algorithm
    # effectively, building a specific decomposition of the covariance matrix

    def _2rank(self, x):
        """compute the measure by which we order samples
        """
        if self.order_by_index is None:
            return np.sum(x) # this is arbitrary, but hopefully it helps to break ties from regular grid spacing
        else:
            return x[self.order_by_index]

    def _2ranks(self, x):
        return np.array([self._2rank(_) for _ in x])

    def _2sorted(self, source_x, source_f=None):
        """sort training data to put it in increasing order
        """
        order = np.argsort(self._2ranks(source_x))
        if source_f is not None:
            source_f = source_f[order]
        return source_x[order], source_f

    def _2neighbors(self, source_x, target_x=None, verbose=False, Verbose=False):
        """identify which elements in source_x (assumed sorted) are neighbors of target_x
        """
        verbose |= Verbose

        if target_x is None: # find neighbor sets within the reference set
            if verbose:
                print('setting target_x = source_x --> finding neighbors within reference set')
            target_x = source_x
            discard_index = 0 # index beyond which we discard possible neighbors
                              # this will discard the current sample, which would also be caught by the exact-match check
                              # will be updated within loop over target_x
        else:
            discard_index = len(source_x) # de facto consider all possible neighbors

        # grab the value by which we first order the reference set
        source_order = self._2ranks(source_x)
        inds = np.arange(len(source_order))

        if verbose:
            print('%d samples in reference set:' % len(source_x))
            if Verbose:
                for X in source_x:
                    print('    %s' % X)

        # iterate over target_x, identifying neighbors for each point
        neighbors = []
        subset = np.empty(len(source_x), dtype=bool)

        if verbose:
            tnd = 0
            num_target = len(target_x)

        for x in target_x:

            if verbose:
                print('processing target %d/%d : %s' % (tnd, num_target, x))
                tnd += 1 # ok to increment this here; we only use it in this print statement

            subset[:] = False # reset the boolean array

            # first, select based on source_order, only looking up to discard_index
            subset[:discard_index] = source_order[:discard_index] <= self._2rank(x)

            if verbose:
                print('    found %d possible neighbors' % np.sum(subset))
                if Verbose:
                    for X in source_x[subset]:
                        print('        %s' % X)

            # make sure there are no exact matches for x within subset
            # find exact matches and exclude them from the subset
            matches = np.all(source_x[subset] == x, axis=1)

            if verbose:
                if np.any(matches):
                    print('    found %d exact matches' % np.sum(matches))
                    if Verbose:
                        for X in source_x[subset][matches]:
                            print('        %s' % X)
                else:
                    print('    no exact matches found!')

            subset[inds[subset][matches]] = False

            if verbose:
                print('    retained %d possible neighbors after excluding exact matches' % np.sum(subset))

            if np.any(subset): # at least one possible sample within the subset
                # compute euclidean distance for all these points
                dist = np.sum((source_x[subset] - x)**2, axis=1)
                order = np.argsort(dist) # order from smallest to largest

                # record the indecies associated with the smallest euclidean distance in the subset
                # order subset by smallest to largest distance, then truncate
                neighbors.append( inds[subset][order][:self.num_neighbors] )

                if verbose:
                    print('    retained %d out of %d with max %d' % \
                        (len(neighbors[-1]), np.sum(subset), self.num_neighbors))
                    if Verbose:
                        for ind, (X, dist) in enumerate(zip(source_x[subset][order], dist[order])):
                            print('        %.6e <-- %s%s' % \
                                (dist, X, '\texcluded' if ind >= self.num_neighbors else '\tneighbor %02d'%ind))

            else: # no neighbors (this should be a special case for at most a few samples)
                if verbose:
                    print('    no neighbors!')
                neighbors.append( [] )

            # increment discard index so we now include the next-biggest element in reference set if needed
            discard_index += 1

        # return
        return neighbors # indecies of neighbors for each point in target_x

    #---

    def _sample2diag(self, x, ref_x, ref_f, verbose=False, safe=True):
        """construct the conditioned distribution for a single sample point
        this should make parallelization easier in the future
        """
        if len(ref_x) == 0: # no neighbors -> just the covariance at this point
            mean = 0.0 # we assume zero-mean process
            diag = self.kernel.cov(x, x)[0]

        else: # run the normal GP conditioning but restricted to the neighbor set
            m, c = Interpolator.condition(self, x, ref_x, ref_f, verbose=verbose)
            mean = m[0]
            diag = c[0,0]

        if safe: # sanity checks
            assert (mean==mean), 'mean is nan\nkernel=%s\nx=%s\nref_x=%s\nref_f=%s' % \
                (self.kernel, x, ref_x, ref_f)
            assert (diag==diag), 'diag is nan\nkernel=%s\nx=%s\nref_x=%s\nref_f=%s' % \
                (self.kernel, x, ref_x, ref_f)
            assert (diag > 0), 'marginal variance is negative!\ndiag=%s\nkernel=%s\nx=%s\nref_x=%s\nref_f=%s' % \
                (diag, self.kernel, x, ref_x, ref_f)

        # return
        return mean, diag

    #---

    def _2diag(self, target_x, source_x, source_f, neighbors, verbose=False):
        """construct the diagonal of the cholesky decomposition and the predicted means
        """
        # FIXME?
        ## can I do the following calculation without the loop in python (which is slow...)?
        ## or at least parallelize this through multiprocessing.pool?
        ## NOTE, an attempt to vectorize via numpy.vectorize resulted in *longer* runtimes, so maybe this is good enough?

        ## this construction was found to be consistently (slightly) faster than a for loop
        ## returns (mean, diag), each of which is a vector with the same length as source_x
        return np.transpose([self._sample2diag(target_x[[ind]], source_x[neighbors[ind]], source_f[neighbors[ind]], verbose=verbose) \
            for ind in range(len(target_x))])

    #--------------------

    def _construct_logprob(
            self,
            source_x,
            source_f,
            logprior=None,
            fixed=None,
            temperature=DEFAULT_TEMPERATURE,
            verbose=False,
            **kwargs # extra kwargs handed to self.loglikelihood within logprob
        ):
        """construct a target function suitable for optimize_kernel and sample_kernel
        """
        # identify neighbor sets once (they won't change)
        source_x, source_f = self._2sorted(source_x, source_f=source_f) # sort the training data
        kwargs['neighbors'] = self._2neighbors(source_x, verbose=verbose) # find neighbors within the training data
                                                                          # and pass these to likelihood
        # delegate
        return Interpolator._construct_logprob(
            self,
            source_x,
            source_f,
            logprior=logprior,
            fixed=fixed,
            temperature=temperature,
            verbose=verbose,
            **kwargs # extra kwargs handed to self.loglikelihood within logprob, which now contains "neighbors"
        )

    #---

    def loglikelihood(self, source_x, source_f, neighbors=None, verbose=False):
        """compute the loglikelihood using the NNGP decomposition of the covariance matrix
        """
        if neighbors is None: # otherwise, we assume everthing has already been sorted and neighbor sets have been identified
            source_x, source_f = self._2sorted(source_x, source_f=source_f) # sort the training data
            neighbors = self._2neighbors(source_x, verbose=verbose)  # find neighbors within the training data

        # compute the combination of 1D Gaussians implicit within the NNGP decomposition
        mean, diag = self._2diag(source_x, source_x, source_f, neighbors, verbose=verbose)

        # loglike is the sum of independnet 1D Gaussians
        return -0.5 * np.sum((mean-source_f)**2 / diag) - 0.5*np.sum(np.log(diag)) - 0.5*len(source_x)*np.log(2*np.pi)

    #--------------------

    def condition(self, target_x, source_x, source_f, verbose=False, Verbose=False):
        """compute conditioned distriution using the NNGP decomposition of the covariance matrix
        """
        verbose |= Verbose

        # first, find neighbors of target_x within source_x
        if verbose:
            print('finding neighbors for %d target_x within %d source_x samples' % (len(target_x), len(source_x)))
            t0 = time.time()

        source_x, source_f = self._2sorted(source_x, source_f=source_f) # sort the training data
        neighbors = self._2neighbors(source_x, target_x=target_x, verbose=Verbose)  # find neighbors

        if verbose:
            print('    time : %.6f sec' % (time.time()-t0))

        # now, assuming that source_x are the reference set, target_x are all (conditionally) independent
        # so we can iterate and evaluate predictions for each target_x separately
        if verbose:
            print('computing predicted means, variances independently')
            t0 = time.time()

        # FIXME!
        ### I may be able to do this more efficiently if I batch jobs based on groups with the same neighbor set
        ### --> avoid repeated inversions of the same source matrix

        mean, diag = self._2diag(target_x, source_x, source_f, neighbors, verbose=Verbose)

        if verbose:
            print('    time : %.6f sec' % (time.time()-t0))

        # format and return
        return mean, np.diag(diag) # cast this to a matrix

    #--------------------

    def compress(self, source_x, source_f, verbose=False, Verbose=False):
        """compress the training set using the NNGP decomposition of the covariance matrix
        """
        # construct covariance matrix
        if verbose:
            print('constructing %d x %d source-source NearestNeighbor covariance matrix with %d neighbors' % \
                (len(source_x), len(source_x), self.num_neighbors))
            t0 = time.time()

        raise NotImplementedError('''
        We need to replace this with the NNGP covariance matrix. We should also be able to speed up the inversion dramatically (i.e., do not use np.linalg.inv but do the inversion by hand)

        cov_src_src = self._x2cov(source_x, source_x, self.kernel, verbose=Verbose)
        inv_cov_src_src = np.linalg.inv(cov_src_src)
''')

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
        """used the compressed representation of the training set to predict the mean at arbitrary points
        """
        # construct covariane matrix
        if verbose:
            print('constructing %d x %d target-source NNGP covariance matrix with %d neighbors' % \
                (len(target_x), len(source_x), self.num_neighbors))
            t0 = time.time()

        raise NotImplementedError('''
        we need to construct the off-diagonal part of the NNGP covariance matrix. The rest of this should follow as-is

        cov_tar_src = self._x2cov(target_x, source_x, self.kernel, verbose=Verbose)
''')

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
