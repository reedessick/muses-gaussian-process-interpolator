"""a module that standarizes logic for reading/writing data from disk as well as constructing GP kernels and interpolators
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

def load(*args, **kwargs):
    """load tabular data from disk
    """
    raise NotImplementedError

#------------------------

def save(*args, **kwargs):
    """write tabular data to disk
    """
    raise NotImplementedError

#-------------------------------------------------

def parse(*args, **kwargs):
    """instantiate a kernel based on a config file
    """
    raise NotImplementedError
