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
   with torch.no_grad():
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
       states[0, 0]
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

Collect Full LSTM Trajectories
------------------------------

This standalone example records both h and c. Prediction readouts should still
use only h. The packed trajectory can be passed directly to sampling, plotting,
and projection helpers as well as the main analysis methods.

.. code-block:: python

   import torch
   from rnntoolkit.adapter import RNNAdapter

   lstm = torch.nn.LSTM(2, 4, batch_first=True).eval()
   adapter = RNNAdapter(lstm)
   inputs = torch.randn(3, 10, 2)
   state = torch.zeros(3, adapter.state_size)
   samples = []
   with torch.no_grad():
       for timestep_input in inputs.unbind(dim=1):
           state = adapter(timestep_input, state)
           samples.append(state)
   packed_states = torch.stack(samples, dim=1)  # [3, 10, 8]
   h, c = adapter.unpack_state(packed_states)   # each [3, 10, 4]

Analyze Packed Or Tuple LSTM States
-----------------------------------

Continuing the preceding example:

.. code-block:: python

   from rnntoolkit import FixedPointFinder, FlowFieldFinder, Linearization

   lin = Linearization(lstm)
   zero_input = torch.zeros(2)
   J_state, J_input = lin.jacobian(zero_input, (h[0, 0], c[0, 0]))
   J_packed, _ = lin.jacobian(zero_input, packed_states[0, 0])
   assert J_state.shape == (8, 8)
   assert torch.allclose(J_state, J_packed)

   finder = FixedPointFinder(lstm, max_iters=100)
   guesses = finder.sample_states(packed_states, n_inits=8, noise_scale=0.1)
   unique_fps, all_fps = finder.find_fixed_points(guesses, zero_input)
   h_fp, c_fp = adapter.unpack_state(unique_fps.xstar)
   # (h_guesses, c_guesses) would also be accepted by find_fixed_points.
   # Inspect qstar before interpreting these short-run candidates as fixed points.

   flow_finder = FlowFieldFinder(lstm, fit_states=(h, c), num_points=5)
   reference_states = packed_states[:, 0]
   reference_inputs = torch.zeros(3, 2)
   nonlinear = flow_finder.find_nonlinear_flow(reference_states, reference_inputs)
   linear = flow_finder.find_linear_flow(
       (h[:, 0], c[:, 0]), reference_inputs, torch.zeros_like(reference_inputs)
   )

Both flows project the entire eight-dimensional [h, c] state. For custom axes,
use shape [2, 8]. To visualize these LSTM trajectories interactively, pass
``packed_states`` to ``visualize`` rather than the tuple (h, c).

Run The Three Flip-Flop Analyses
--------------------------------

From the repository root:

.. code-block:: bash

   python examples/analysis.py

The script reads flip_flop_rnn.pth, flip_flop_gru.pth, and flip_flop_lstm.pth
from its own directory, selecting RNNModel, GRUModel, and LSTMModel respectively.
All checkpoints must match the example's three input/output channels and
16 hidden units. LSTM analysis states have width 32; RNN/GRU states have width 16.
The LSTM model's ordinary forward method takes native (h0, c0) but returns
only hidden trajectories. The script's collect_trajectories helper records
full cell trajectories through one-step updates.

Each architecture produces six figures under examples/analysis_results/<name>:

#. Trial predictions.
#. Fixed points and trajectories in three-dimensional PCA.
#. Nonlinear zero-input flow fields.
#. Fixed points colored by full-state eigenvalue stability.
#. Linear flow fields around fixed points.
#. Perturbation alignment with unstable modes.

The script uses the noninteractive Agg backend and allows 35,000 optimization
iterations per architecture. There are no CLI flags. For a short execution
check, call run_analysis from Python with an explicit matching model_class:

.. code-block:: python

   # In a Python session with examples/ on sys.path:
   from pathlib import Path
   from analysis import EXAMPLE_DIR, run_analysis
   from model import LSTMModel

   unique_fps, all_fps = run_analysis(
       EXAMPLE_DIR / "flip_flop_lstm.pth",
       max_iters=2,
       output_dir=Path("/tmp/lstm-analysis-check"),
       model_class=LSTMModel,
   )

A short run checks execution, not convergence. Inspect qstar before drawing
fixed-point or stability conclusions. Checkpoints and training are separate
from analysis; running analysis does not train a model.
