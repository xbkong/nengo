"""A customizable configuration system for setting default parameters and
backend-specific info.

The idea here is that a backend can create a Config and ConfigItems to
define the set of parameters that their backend supports.
Parameters are done as Python descriptors, so backends can also specify
error checking on those parameters.

A good writeup on descriptors (which has an example similar to Parameter)
can be found at
http://nbviewer.ipython.org/urls/gist.github.com/ChrisBeaumont/5758381/raw/descriptor_writeup.ipynb  # noqa

"""

import inspect

from nengo.exceptions import ConfigError
from nengo.params import is_param
from nengo.utils.compat import itervalues
from nengo.utils.threading import ThreadLocalStack


class Config(object):
    """Configures network-level behavior and backend specific parameters.

    Every ``Network`` contains an associated ``Config`` object which can
    be manipulated to change overall network behavior, and to store
    backend specific parameters. Subnetworks inherit the ``Config`` of
    their parent, but can be manipulated independently.
    The top-level network inherits ``nengo.toplevel_config``.

    Attributes
    ----------
    params : dict
        Maps configured classes and instances to their ``ClassParams``
        or ``InstanceParams`` object.

    Example
    -------
    >>> class A(object): pass
    >>> inst = A()
    >>> config = Config(A)
    >>> config[A].set_param('amount', Parameter(default=1))
    >>> print(config[inst].amount)
    1
    >>> config[inst].amount = 3
    >>> print(config[inst].amount)
    3
    >>> print(config[A].amount)
    1
    """

    context = ThreadLocalStack(maxsize=100)  # static stack of Config objects

    def __init__(self):
        self.params = {}

    def __enter__(self):
        Config.context.append(self)
        return self

    def __exit__(self, dummy_exc_type, dummy_exc_value, dummy_tb):
        if len(Config.context) == 0:
            raise ConfigError("Config.context in bad state; was empty when "
                              "exiting from a 'with' block.")

        config = Config.context.pop()

        if config is not self:
            raise ConfigError("Config.context in bad state; was expecting "
                              "current context to be '%s' but instead got "
                              "'%s'." % (self, config))

    def __getitem__(self, key):
        class Accessor(object):
            def has(_, item):
                if item in self.params.get(key, {}):
                    return True

                # If key is a class return a superclass's ClassParams
                if inspect.isclass(key):
                    for cls in key.__mro__:
                        if item in self.params.get(cls, {}):
                            return True

                return False

            def __getattr__(_, item):
                # If we have the exact thing, we'll just return it
                if item in self.params.get(key, {}):
                    return self.params[key][item]

                # If key is a class return a superclass's ClassParams
                if inspect.isclass(key):
                    for cls in key.__mro__:
                        if item in self.params.get(cls, {}):
                            return self.params[cls][item]

                    return getattr(key, item).default
                    # raise KeyError("No entry for this class")
                else:
                    raise KeyError("No entry for this instance")

            def __delattr__(this, item):
                if not this.has(item):
                    raise KeyError("No such attribute")

                if item in self.params.get(key, {}):
                    del self.params[key][item]

                if inspect.isclass(key):
                    for cls in key.__mro__:
                        if item in self.params.get(cls, {}):
                            del self.params[cls][item]

            def __setattr__(_, item, value):
                if not inspect.isclass(key) and is_param(getattr(type(key), item)):
                    raise ConfigError(
                        "Cannot configure the built-in parameter '%s' on an instance "
                        "of '%s'. Please get the attribute directly from the object."
                        % (key, type(key).__name__))

                self.params.setdefault(key, {})[item] = value

        return Accessor()

    def __repr__(self):
        classes = [key.__name__ for key in self.params if inspect.isclass(key)]
        return "<%s(%s)>" % (self.__class__.__name__, ', '.join(classes))

    def __str__(self):
        return "\n".join(str(v) for v in itervalues(self.params))

    # @staticmethod
    # def all_defaults(nengo_cls=None):
    #     """Look up all of the default values in the current context.

    #     Parameters
    #     ----------
    #     nengo_cls : class, optional
    #         If specified, only the defaults for a particular class will
    #         be returned. If not specified, the defaults for all configured
    #         classes will be returned.

    #     Returns
    #     -------
    #     str
    #     """
    #     lines = []
    #     if nengo_cls is None:
    #         all_configured = set()
    #         for config in Config.context:
    #             all_configured.update(key for key in config.params
    #                                   if inspect.isclass(key))
    #         lines.extend([Config.all_defaults(key) for key in all_configured])
    #     else:
    #         lines.append("Current defaults for %s:" % nengo_cls.__name__)
    #         for attr in dir(nengo_cls):
    #             desc = getattr(nengo_cls, attr)
    #             if is_param(desc) and desc.configurable:
    #                 val = Config.default(nengo_cls, attr)
    #                 lines.append("  %s: %s" % (attr, val))
    #     return "\n".join(lines)

    @staticmethod
    def default(nengo_cls, param):
        """Look up the current default value for a parameter.

        The default is found by going through the config stack, from most
        specific to least specific. The network that an object is in
        is the most specific; the top-level network is the least specific.
        If no default is found there, then the parameter's default value
        is returned.
        """
        # Get the descriptor
        desc = getattr(nengo_cls, param)
        if not desc.configurable:
            raise ConfigError("Unconfigurable parameters have no defaults. "
                              "Please ensure you are not using the 'Default' "
                              "keyword with an unconfigurable parameter.")

        for config in reversed(Config.context):
            if config[nengo_cls].has(param):
                return getattr(config[nengo_cls], param)

        # Otherwise, return the param default
        return desc.default
