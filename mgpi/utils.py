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

def _factory(klass):
    """discover and return all the subclasses of a particular class
    """
    ans = dict()
    for obj in klass.__subclasses__():
        ans[obj.__name__] = obj
        ans.update(_factory(obj))
    return ans

#-------------------------------------------------

def subdivide_data(source_x, source_f, **kwargs):
    """divide the data into subsets for testing and training
    """
    raise NotImplementedError
