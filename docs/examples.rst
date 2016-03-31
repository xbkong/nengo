********
Examples
********

Nengo creates these models using the principles of the
`Neural Engineering Framework <http://compneuro.uwaterloo.ca/research/nef.html>`_.
The first set of examples explains these three principles:
*representation*, *transformation*, and *dynamics*.

For a summary of these principles, see the following example:

.. toctree::
   :maxdepth: 1

   examples/nef_summary

The following examples give a more detailed presentation:

.. toctree::
   :maxdepth: 2

   representation
   transformation
   dynamics

Putting these three principles together allows us to scale
these examples up to larger networks that do more complex functions.
Below are some of these complex functions,
as well as other examples that we hope will be helpful
as reference when building your own Nengo models.

Nodes
=====

.. toctree::
   :maxdepth: 1

   examples/delay_node

Ensembles
=========

.. toctree::
   :maxdepth: 1

   examples/tuning_curves
   examples/izhikevich

Connections
===========

.. toctree::

   examples/inhibitory_gating

Learning
========

.. toctree::
   :maxdepth: 1

   examples/learn_communication_channel
   examples/learn_square
   examples/learn_product
   examples/learn_unsupervised

Networks
========

.. toctree::
   :maxdepth: 1

   examples/ensemble_array
   examples/matrix_multiplication
   examples/basal_ganglia
   examples/integrator_network

Semantic Pointer Architecture
=============================

.. toctree::
   :maxdepth: 1

   examples/convolution
   examples/question
   examples/question_control
   examples/question_memory
   examples/spa_sequence
   examples/spa_sequence_routed
   examples/spa_parser

Extending Nengo
===============

.. toctree::
   :maxdepth: 1

   examples/rectified_linear
