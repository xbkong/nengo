
# coding: utf-8

# # Context matters, membership doesn't
# 
# When you create a Nengo object
# and leave the parameters as their default values,
# we determine the default values dynamically
# based on `Config` objects.
# Every `Network` has an associated `Config` object,
# and you can make new `Config` objects for additional flexibility.
# 
# Nengo keeps track of the current context.
# The following example works based on the context.

# In[ ]:

import nengo


# In[ ]:

with nengo.Network() as model:
    model.config[nengo.Ensemble].radius = 2.0
    subnet = nengo.Network()

    with subnet:
        a = nengo.Ensemble(10, 1)
        print(a.radius)


# The radius of `a` is `2.0`
# because the current context includes both
# `subnet` and `model`.
# `subnet` does not change the default value of the radius,
# but `model` does, so it uses the `model` default of `2.0`.
# 
# Here's a similar example.

# In[ ]:

with nengo.Network() as model:
    model.config[nengo.Ensemble].radius = 2.0
    subnet = nengo.Network()

with subnet:
    a = nengo.Ensemble(10, 1)
    print(a.radius)


# While this example looks nearly identical,
# the difference is that `a` is created
# in the context of `subnet` only;
# `model` is not part of the config context.
# Because of that, when `a` is created,
# Nengo sees that no default is set in `subnet`
# and uses the global default value of `1.0`.
# 
# This may seem counterintuitive
# since `subnet` is a member of `model`;
# it's stored as a sub-network of the `model` network.
# However, Nengo objects are not aware of their parents.
# This allows `subnet` to be used the same way
# whether it's the top-level network
# or whether it's nested twenty layers deep,
# but it also means that we can't set defaults
# based on network membership.

# ## Context details
# 
# The configuration context is stored in the
# `nengo.Config` class as `nengo.Config.context`.
# `context` is a thread-local list.
# We add new contexts to the end of that list
# at the start of `with` blocks,
# and pop contexts off of that list
# at the end of `with` blocks.

# In[ ]:

# No context
len(nengo.Config.context)


# In[ ]:

# Model context
with model:
    print(len(nengo.Config.context))
    print(nengo.Config.context[0] is model.config)


# In[ ]:

# Subnet, Model context
with model:
    with subnet:
        print(len(nengo.Config.context))
        print(nengo.Config.context[0] is model.config)
        print(nengo.Config.context[1] is subnet.config)


# If you are not sure what context you're in,
# but you want to know what the defaults are
# in the current context,
# use `nengo.Config.all_defaults`.
# You can optionally pass in the type that you're interested in.

# In[ ]:

with model:
    with subnet:
        print(nengo.Config.all_defaults(nengo.Ensemble))


# In[ ]:

with subnet:
    print(nengo.Config.all_defaults(nengo.Ensemble))


# Note above that the radius changes
# despite the fact that both situations occur
# in the context of `subnet`!
# Configuration outside matters if no default
# has been set in the immediate context.
