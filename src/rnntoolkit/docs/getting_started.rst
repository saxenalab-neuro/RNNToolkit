Getting Started
===============

Installation
------------

Use Python 3.11+ for the current source tree (which uses typing.Self).
Install runtime dependencies explicitly; current package metadata is incomplete.
From the repository root:

.. code-block:: bash

   pip install torch numpy scikit-learn matplotlib pygame
   pip install -e .

Create And Run An RNN
---------------------

The tools support single-layer, unidirectional RNN/GRU/LSTM modules without
LSTM projections. This example uses batch_first=True for trajectories shaped
[batch, time, features]; either layout is supported by the adapter.
For a GRU, replace nn.RNN with nn.GRU and omit nonlinearity.

.. code-block:: python

   import torch

   rnn = torch.nn.RNN(
       input_size=2,
       hidden_size=4,
       nonlinearity="tanh",
       batch_first=True,
   )
   inputs = torch.randn(8, 30, 2)
   h0 = torch.zeros(1, 8, 4)
   with torch.no_grad():
       states, final_state = rnn(inputs, h0)

Run An Analysis
---------------

The top-level package exports the main analysis classes. For example, compute
local Jacobians around one input and hidden state:

.. code-block:: python

   from rnntoolkit import Linearization

   linearization = Linearization(rnn)
   recurrent_jacobian, input_jacobian = linearization.jacobian(
       inputs[0, 0], states[0, 0]
   )

LSTM States
-----------

LSTMs need both h and c. Their packed analysis state is [h, c] with width
2 * hidden_size. Main analysis methods accept packed tensors or native tuples;
sampling, projection helpers, and visualizer trajectories require packed tensors.
See :doc:`user_guide` for the complete contract and :doc:`examples` for rollout.

Next Steps
----------

See :doc:`user_guide` for conceptual workflows and :doc:`examples` for
copy-pastable fixed-point, flow-field, and visualization snippets.
