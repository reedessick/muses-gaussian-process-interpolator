"""a module that standarizes logic for reading/writing data from disk as well as constructing GP kernels and interpolators
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

from configparser import ConfigParser
from . import mgpi

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

def parse_table(*args, **kwargs):
    """load data based on a config file
    """
    raise NotImplementedError

#-------------------------------------------------

def _factory(klass):
    """discover and return all the subclasses of a particular class
    """
    ans = dict()
    for obj in klass.__subclasses__():
        ans[obj.__name__] = obj
        ans.update(_factory(obj))
    return ans

#------------------------

__KERNEL_TYPE_NAME__ = 'type'

def parse_kernel_section(config, section, verbose=False):
    """parse the individual section within a kernel INI file.
Each section in the INI should follow the format

[Name]
%s = Kernel # change this to match the name of the type of kernel you want to instantiate
arg0 = ...    # list the values for the arguments that will be passed to Kernel.__init__ in the order they should be passed
arg1 = ...
arg2 = ...
    """ % __KERNEL_TYPE_NAME__

    # first, figure out which object we need to instantiate
    assert config.has_option(section, __KERNEL_TYPE_NAME__), 'could not find "type" in section=%s' % section

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
