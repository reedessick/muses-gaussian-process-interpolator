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
__ASCII_PRIOR_NAME__ = 'prior' # used to identify options that are associated with prior limits on data in the table
__ASCII_DOWNSAMPLE_NAME__ = 'downsample' # used to specify how we should reduce the data's size

def parse_ascii_data(config, section, verbose=False):
    """parse dat, txt, or csv tabular data
    """
    options = config.options(section)

    # load the path
    for option in [__ASCII_PATH_NAME__, __ASCII_X_NAME__, __ASCII_F_NAME__]:
        assert option in options, 'could not find %s in section=%s' % (option, section)

    path = config.get(section, __ASCII_PATH_NAME__)
    xcols = config.get(section, __ASCII_X_NAME__).split()
    fcol = config.get(section, __ASCII_F_NAME__)

    # sanity check
    assert len(xcols) == len(set(xcols)), 'cannot have repeated values in %s=%s' % (__ASCII_X_NAME__, ', '.join(xcols))
    assert fcol not in xcols, 'cannot have %s=%s in %s=%s' % (__ASCII_F_NAME__, fcol, __ASCII_X_NAME__, ', '.join(xcols))

    # load priors
    priors = dict()
    for option in options:
        opt = option.split()
        if opt[0] == __ASCII_PRIOR_NAME__: # treat this as a prior
            priors[opt[1]] = [float(_) for _ in config.get(section, option).split()]

    if verbose:
        print('    loading ascii data from: '+path)
        print('    source_x\n        %s' % ('\n        '.join(xcols)))
        print('    source_f\n        %s' % fcol)
        if priors:
            print('    priors\n        %s' % ('\n        '.join('%.3e <= %s <= %.3e' % (m, c, M) for c, (m,M) in priors.items())))

    # load the data
    data = load_ascii_data(path)

    if verbose:
        print('    found %d samples' % len(data))

    # check that we have the columns we need
    for col in xcols + [fcol]:
        assert col in data.dtype.names, 'required column=%s not present!' % col

    # apply priors
    for col in data.dtype.names:
        if col.lower() in priors:
            m, M = priors[col.lower()]
            keep = (m <= data[col]) * (data[col] <= M)
            if verbose:
                print('    retaining %d samples after imposing: %.3e <= %s <= %.3e' % (np.sum(keep), m, col, M))
            data = data[keep]

    # downsample the data
    if __ASCII_DOWNSAMPLE_NAME__ in options:
        downsample = config.getint(section, __ASCII_DOWNSAMPLE_NAME__)
        if verbose:
            print('    downsampling data to retain 1 out of every %d samples' % downsample)
        data = data[::downsample]

        if verbose:
            print('        retained %d samples' % len(data))

    # now extract
    source_x = np.transpose([data[col] for col in xcols])
    source_f = data[fcol]

    # return
    return (source_x, source_f), (xcols, fcol)

#-----------

def load_ascii_data(path, verbose=False):
    if verbose:
        print('loading: '+path)
    return np.genfromtxt(
        path,
        names=True,
        delimiter=',' if any(path.endswith(_) for _ in ['csv', 'csv.gz']) else None,
    )

#-----------

def save_ascii_data(path, source_x, source_f, xcols=None, fcol='f', verbose=False):
    """write tabular data into an ascii file
    """
    nsmp, ndim = source_x.shape

    if xcols is None:
        xcols = ['x%d'%dim for dim in range(ndim)]
    if verbose:
        print('writing %d samples with dimension (%d+1) to: %s' % (nsmp, ndim, path))

    delimiter = ',' if any(path.endswith(_) for _ in ['csv', 'csv.gz']) else ' '

    np.savetxt(
        path,
        np.transpose([source_x[:,dim] for dim in range(ndim)] + [source_f]),
        header=delimiter.join(list(xcols)+[fcol]),
        comments='',
        delimiter=delimiter,
    )

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

__INTERPOLATOR_NAME__ = 'Interpolator'
__INTERPOLATOR_TYPE_NAME__ = 'type'
__INTERPOLATOR_KERNEL_NAME__ = 'kernel'

def parse_interpolator(path, verbose=False):
    """load an interpolator based on a config file.
There should be a single section called %(name)s that specifies the interpolator. This section should have the format
[%(name)s]
%(type)s = Interpolator
kernel = sec1 sec2 # can repeat multiple sections to build a CombinedKernel
kwarg0 = ...       # the rest of the options will be passed to the Interpolator's instantiate
kwarg1 = ...
kwarg2 = ...
    """ % {'name':__INTERPOLATOR_NAME__, 'type':__INTERPOLATOR_TYPE_NAME__}

    if verbose:
        print('reading interpolator config from: '+path)
    config = ConfigParser()
    config.read(path)

    # set up the options for the interpolator
    assert config.has_section(__INTERPOLATOR_NAME__), 'interpolator config must have section [%s]' % __INTERPOLATOR_NAME__
    options = config.options(__INTERPOLATOR_NAME__)

    assert __INTERPOLATOR_TYPE_NAME__ in options, \
        'cannot find %s in section=%s' % (__INTERPOLATOR_NAME__, __INTERPOLATOR_TYPE_NAME__)

    # grab the instantiator from dynamic list of implemented kernels
    interp_type = config.get(__INTERPOLATOR_NAME__, __INTERPOLATOR_TYPE_NAME__)
    options.remove(__INTERPOLATOR_TYPE_NAME__)

    # iterate over kernels and load each section in turn
    if verbose:
        print('parsing kernel')

    kernel = []
    for name in config.get(__INTERPOLATOR_NAME__, __INTERPOLATOR_KERNEL_NAME__).split():
        assert config.has_section(name), 'can not find section=%s' % name
        try:
            kernel.append(parse_kernel_section(config, name, verbose=verbose))
        except:
            warnings.Warn('could not parse section=%s. Skipping...' % name)

    assert kernel, 'could not find any kernels within: '+path
    if len(kernel) > 1:
        kernel = mgpi.CombinedKernel(*kernel)
    else:
        kernel = kernel[0]

    options.remove(__INTERPOLATOR_KERNEL_NAME__)

    # get the rest of the options as kwargs
    kwargs = dict()
    for option in options:
        try:
            val = config.getint(__INTERPOLATOR_NAME__, option)
        except ValueError:
            try:
                val = config.getfloat(__INTERPOLATOR_NAME__, option)
            except ValueError:
                try:
                    val = config.getboolean(__INTERPOLATOR_NAME__, option)
                except ValueError:
                    val = config.get(__INTERPOLATOR_NAME__, option)
        kwargs[option] = val

    # instantiate the interpolator
    if verbose:
        print('instantiating interpolator')
        print('  %s' % interp_type)
        print('  %s' % kernel)
        for key, val in kwargs.items():
            print('  %s = %s' % (key, val))

    interp = _factory(mgpi.Interpolator)[interp_type](kernel, **kwargs)

    # return
    return interp
