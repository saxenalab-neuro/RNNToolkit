Examples
========

The examples below are short templates intended for scripts or notebooks.

Collect A Trajectory
--------------------

.. code-block:: python

   import torch

   rnn = torch.nn.RNN(2, 4, nonlinearity="tanh", batch_first=True)
   inputs = torch.randn(8, 30, 2)
   h0 = torch.zeros(1, 8, 4)
   states, _ = rnn(inputs, h0)

Find Fixed Points
-----------------

.. code-block:: python

   from rnntoolkit import FixedPointFinder

   finder = FixedPointFinder(rnn, max_iters=1000, verbose=False)
   initial_states = states.reshape(-1, states.shape[-1])[::20]
   fixed_input = torch.zeros(inputs.shape[-1])
   unique_fps, all_fps = finder.find_fixed_points(initial_states, fixed_input)
   print(unique_fps.n)

Linearize Around A State
------------------------

.. code-block:: python

   from rnntoolkit import Linearization

   linearization = Linearization(rnn)
   jacobian, input_jacobian = linearization.jacobian(
       inputs[0, 0], states[0, 0]
   )
   real_parts, imaginary_parts, eigenvectors = linearization.eigendecomposition(
       inputs[0, 0], states[0, 0]
   )

Compute A Nonlinear Flow Field
------------------------------

.. code-block:: python

   from rnntoolkit import FlowFieldFinder

   flow_finder = FlowFieldFinder(
       rnn,
       num_points=25,
       x_offset=2,
       y_offset=2,
       fit_states=states,
   )
   fields = flow_finder.find_nonlinear_flow(states[:, :1], inputs[:, :1])
   first_field = fields[0]

Open The Interactive Visualizer
-------------------------------

.. code-block:: python

   from rnntoolkit import FlowFieldVisualizer

   axes = torch.eye(states.shape[-1])
   visualizer = FlowFieldVisualizer(
       rnn,
       num_points=25,
       axes=axes,
       axis_labels=["unit 1", "unit 2", "unit 3", "unit 4"],
       flow_type="nonlinear",
   )
   visualizer.visualize(inputs, states)
