"""a module that standarizes logic for reading/writing data from disk as well as constructing GP kernels and interpolators
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

from configparser import ConfigParser

import numpy as np

# non-standard libraries
from . import mgpi

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

__FILE_TYPE_NAME__ = 'type' # the protected option name that specifies the type of file within the section

def parse_table(path, section=None, verbose=False):
    """load data based on a config file
    """
    if verbose:
        print('reading tabular data from: '+path)
    config = ConfigParser()
    config.read(path)

    # figure out which section we're supposed to read
    if section is None:
        section = config.sections()
        assert len(section) == 1, 'must specify a section when multiple exist within %s\n%s' % \
            (path, '\n'.join(section))
        section = section[0]

    if verbose:
        print('    reading section='+section)

    # now parse that section
    assert config.has_option(section, __FILE_TYPE_NAME__), 'could not find %s in section=%s' % (__FILE_TYPE_NAME__, section)
    filetype = config.get(section, __FILE_TYPE_NAME__)

    if filetype == 'ascii':
        return parse_ascii_data(config, section, verbose=verbose)

    elif filetype == 'stellarcollapse':
        return parse_stellar_collapse_data(config, section, verbose=verbose)

    else:
        raise ValueError('could not understand %s=%s' % (__FILE_TYPE_NAME__, filetype))

#------------------------

__ASCII_PATH_NAME__ = 'path' # the protected option name that specifies the path of the file within the section
__ASCII_X_NAME__ = 'x_columns' # protected option specifying the names and order of the x-columns to used within interpolator
__ASCII_F_NAME__ = 'f_column'  # protected option specifying the target function which we will emulate

def parse_ascii_data(config, section, verbose=False):
    """parse dat, txt, or csv tabular data
    """
    # load the path
    for option in [__ASCII_PATH_NAME__, __ASCII_X_NAME__, __ASCII_F_NAME__]:
        assert config.has_option(section, option), 'could not find %s in section=%s' % (option, section)
    path = config.get(section, __ASCII_PATH_NAME__)
    xcols = config.get(section, __ASCII_X_NAME__).split()
    fcol = config.get(section, __ASCII_F_NAME__)

    # sanity check
    assert len(xcols) == len(set(xcols)), 'cannot have repeated values in %s=%s' % (__ASCII_X_NAME__, ', '.join(xcols))
    assert fcol not in xcols, 'cannot have %s=%s in %s=%s' % (__ASCII_F_NAME__, fcol, __ASCII_X_NAME__, ', '.join(xcols))

    if verbose:
        print('    loading ascii data from: '+path)
        print('    source_x\n        %s' % ('\n        '.join(xcols)))
        print('    source_f\n        %s' % fcol)

    # load the data
    data = np.genfromtxt(
        path,
        names=True,
        delimiter=',' if any(path.endswith(_) for _ in ['csv', 'csv.gz']) else None,
    )

    # now extract and format the data
    for col in xcols + [fcol]:
        assert col in data.dtype.names, 'required column=%s not present!' % col

    source_x = np.transpose([data[col] for col in xcols])
    source_f = data[fcol]

    # return
    return (source_x, source_f), (xcols, fcol)

#-----------

def save_ascii_data(*args, **kwargs):
    """write tabular data into an ascii file
    """
    raise NotImplementedError

#------------------------

def parse_stellarcollapse_data(config, section, verbose=False):
    """parse tabular data from HDF structures defined by: https://stellarcollapse.org/equationofstate.html
    """
    raise NotImplementedError

#-----------

def save_stellarcollapse_data(*args, **kwargs):
    """write tabular data into HDF structures defined by: https://stellarcollapse.org/equationofstate.html
    """
    raise NotImplementedError

#-------------------------------------------------

__KERNEL_TYPE_NAME__ = 'type' # the protected option name that specifies the type of kernel within the section

def parse_kernel_section(config, section, verbose=False):
    """parse the individual section within a kernel INI file.
Each section in the INI should follow the format

[Name]      # this is essentially ignored and can be anything
%s = Kernel # change this to match the name of the type of kernel you want to instantiate
arg0 = ...  # list the values for the arguments that will be passed to Kernel.__init__ in the order they should be passed
arg1 = ...
arg2 = ...
    """ % __KERNEL_TYPE_NAME__

    # first, figure out which object we need to instantiate
    assert config.has_option(section, __KERNEL_TYPE_NAME__), 'could not find %s in section=%s' % (__KERNEL_TYPE_NAME__, section)

    # grab the instantiator from dynamic list of implemented kernels
    kernel = _factory(mgpi.Kernel)[config.get(section, __KERNEL_TYPE_NAME__)]

    # now parse the options
    options = config.options(section)
    options.remove(__KERNEL_TYPE_NAME__)
    args = []
    for option in options:
        try:
            val = config.getint(section, option)
        except ValueError:
            try:
                val = config.getfloat(section, option)
            except ValueError:
                raise ValueError('could not parse option=%s in section=%s' % (option, section))

        args.append(val)

    # instantiate the object
    kernel = kernel(*args)

    # report
    if verbose:
        print('    section=%s -> %s' % (section, kernel))

    # return
    return kernel

#------------------------

def parse_kernel(path, verbose=False):
    """load a kernel based on a config file.
Multiple kernels can be defined in the same INI as separate sections. In this case, we return a CombinedKernel constituting the sum of the separate kernels.
Each section in the INI should follow the format

[Name]
type = Kernel # change this to match the name of the type of kernel you want to instantiate
arg0 = ...    # list the values for the arguments that will be passed to Kernel.__init__ in the order they should be passed
arg1 = ...
arg2 = ...
    """

    if verbose:
        print('reading kernel config from: '+path)
    config = ConfigParser()
    config.read(path)

    # iterate through sections and parse each in turn
    kernels = []
    for name in config.sections():
        try:
            kernels.append(parse_kernel_section(config, name, verbose=verbose))
        except:
            warnings.Warn('could not parse section=%s. Skipping...' % name)

    # sanity check results
    assert kernels, 'could not find any kernels within: '+path

    # return
    if len(kernels) > 1:
        return mgpi.CombinedKernel(*kernels)
    else:
        return kernels[0]