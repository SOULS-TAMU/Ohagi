Export and load a JAX graph
===========================

This example isolates the deployment interface from a training run. In a
trained model, the graph captures learned parameters rather than the fixed
coefficients used here.

.. literalinclude:: deployment.py
   :language: python
   :linenos:

Download: :download:`deployment.py <deployment.py>`.
