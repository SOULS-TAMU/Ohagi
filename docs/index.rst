OHAGI
=====

One shared JAX engine for constrained neural training and inference.

This guide describes version **1.0.0**. Start with the runnable example,
then use the guide and API reference for your own problem.

Quick overview
--------------

OHAGI composes the shared learning workflow:

1. Receive a problem and prepared PURIN projection.
2. Wrap a functional JAX, custom or optional Flax backbone.
3. Evaluate the loss and update explicit model/optimizer state.
4. Export a JAX prediction graph and reuse the metadata-loaded predictor.

The :doc:`training example <examples/quickstart>` needs no frontend. The
:doc:`deployment example <examples/deployment>` demonstrates the common
single-instance inference interface.

.. toctree::
   :maxdepth: 2
   :caption: Start here

   INSTALL
   examples/index
   user-guide/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index
   ECOSYSTEM
   MIGRATION
   citations
   license
   contributing
   changelog
   BUILD
   PUBLISHING

Indices and search
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
