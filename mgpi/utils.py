"""basic utilities
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import numpy as np

#-------------------------------------------------

def seed(s, verbose=False):
    """set numpy's random seed
    """
    if verbose:
        print('setting numpy.seed=%d' % args.seed)
    np.seed(args.seed)

#-------------------------------------------------

def factory(klass):
    """discover and return all the subclasses of a particular class
    """
    ans = dict()
    for obj in klass.__subclasses__():
        ans[obj.__name__] = obj
        ans.update(factory(obj))
    return ans

#-------------------------------------------------

def construct_logprior(names, bounds, fixed):
    """define a function on-the-fly that will act as an appropriate logprior
    """
    names = [name for name in names if (name not in fixed)]
    for key in bounds.keys():
        assert key in names, 'cannot set prior for unknown parameter: '+key

    # convert to index instead of parameter name
    bounds = dict((names.index(key), val) for key, val in bounds.items())

    # define the function
    def logprob(params):
        for ind, (m, M) in bounds.items():
            if (params[ind] < m) or (M < params[ind]):
                return -np.infty
        return 0.0

    # return
    return logprob

#-------------------------------------------------

def subdivide_data(source_x, source_f, frac=0.5, verbose=False):
    """divide the data into subsets for testing and training. We randomly assign "frac" of the data \
into the train data set and "1 - frac" into the test data set.
    """
    # figure out the sizes of the subsets
    num = len(source_x)
    num_train = int(round(frac*num, 0))

    if verbose:
        print('dividing data set of %d samples into %d training samples and %d testing samples' % \
            (num, num_train, num-num_train))

    assert (num_train > 0), 'will not partion data into training set with zero samples'
    assert (num > num_train), 'will not partion data into training set with zero samples'

    # generate random subsamples
    inds = np.arange(num)
    np.random.shuffle(inds) # randomly shuffle order

    # return the subsets
    #       training data                                             test data
    return (source_x[inds[:num_train]], source_f[inds[:num_train]]), (source_x[inds[num_train:]], source_f[inds[num_train:]])
