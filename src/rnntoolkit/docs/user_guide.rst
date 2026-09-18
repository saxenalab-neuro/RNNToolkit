User Guide
==========

Supported Networks
------------------

The concrete tools use ``RNNAdapter`` and support single-layer, unidirectional
``torch.nn.RNN``, ``torch.nn.GRU``, and ``torch.nn.LSTM`` modules. Both
``batch_first`` layouts work. Multilayer modules, bidirectional modules,
projected LSTMs, and ``RNNCell``/``GRUCell``/``LSTMCell`` are not supported by
this adapter. Custom dynamics can implement the finder base-class interfaces.

State Representation
--------------------

Let H be hidden_size, I be input_size, and D be the full analysis state width.
For RNN/GRU, D = H and the state is h. For LSTM, D = 2H and the packed state
is the concatenation [h, c] along the last dimension. Native LSTM pairs have
matching hidden and cell shapes. Never omit c or reset it between analysis
steps: it affects the subsequent hidden-state dynamics.

.. list-table:: Accepted LSTM representations
   :header-rows: 1
   :widths: 65 15 20

   * - Entry point
     - Packed
     - (h, c) tuple
   * - Linearization: jacobian, eigendecomposition, forward/call
     - Yes
     - Yes
   * - FixedPointFinder.find_fixed_points
     - Yes
     - Yes
   * - FlowFieldFinder: both flow methods and constructor fit_states
     - Yes
     - Yes
   * - Sampling, projection, and analysis plotting helpers
     - Yes
     - No
   * - FlowFieldVisualizer.visualize trajectory states
     - Yes
     - No
   * - RNNAdapter.step/call and unpack_state
     - Yes
     - No
   * - RNNAdapter.pack_state
     - No
     - Yes

All next-state and fixed-point outputs are packed. ``unpack_state`` returns
(h, c) views for LSTMs. ``is_packed`` checks tensor type and final width for
LSTMs and always returns True for RNN/GRU; it is not full input validation.
A False result does not establish that an object is a valid unpacked pair.

Preparing Trajectories
----------------------

Record representative trajectories, usually [batch, time, D]. Sampling and
flow methods flatten leading dimensions into samples; linearization uses one
reference input [I] and state [D]. Keep reference inputs aligned with the states
where you want to evaluate F(input, state). For a recorded post-update state,
its following actual transition uses the next timestep's input. Zero-input
analyses deliberately hold the external drive at zero instead.

PyTorch LSTM sequence output contains hidden trajectories and the final cell
state, not a complete cell trajectory. Record both components by stepping the
module through the sequence. See :doc:`examples` and
``examples/analysis.py:collect_trajectories``. Pack before sampling so h and c
from the same timestep remain paired.

The adapter's step interface accepts [I]/[D] or [N, I]/[N, D] and adds native
sequence/layer dimensions internally. Do not add those dimensions yourself.
Inputs and states must match the model dtype/device. The documented PCA
workflow uses detached float32 CPU tensors because projection uses scikit-learn;
the adapter itself does not move or detach data and preserves autograd.

Linearization
-------------

``jacobian(input, state)`` returns J_state [D, D] and J_input [D, I]. For an
LSTM this includes all four hidden/cell derivative blocks. Matrix dimensions
are retained for one-dimensional states and inputs.

``forward(input, state, delta_input, delta_state)`` evaluates the affine map
F(input, state) + J_state delta_state + J_input delta_input. State perturbations
may be packed tensors or LSTM (delta_h, delta_c) tuples. Batched perturbations
return [N, D]; a single perturbation returns [D].

``eigendecomposition(state)`` evaluates at zero input and returns real parts,
imaginary parts, and column eigenvectors. For a nonzero reference input, apply
``torch.linalg.eig`` to ``jacobian(input, state)[0]``. A discrete-time fixed
point is strictly locally stable when every eigenvalue magnitude is below one.
Magnitudes greater than one imply instability; unit magnitudes are inconclusive
at linear order. Stability concerns the full state, not just an LSTM h block.

Fixed Points
------------

``find_fixed_points`` accepts guesses [N, D] or a single [D], with either a
shared constant input [I] or inputs [N, I] held constant for each guess. LSTM
pairs are packed internally. The objective is

.. math::

   q(s) = \tfrac{1}{2}\lVert F(u,s)-s\rVert^2.

The LSTM objective includes both hidden and cell residuals. The returned pair
is (unique_fps, all_fps). State fields xstar, x_init, and F_xstar remain packed;
x_init retains original guesses and qstar is evaluated at the returned xstar.
Uniqueness and distance filtering operate on the complete packed state.

Inspect qstar: iteration limits or a small change in q do not establish
convergence. Filtering can leave an empty collection. Collections support
indexing, filtering, transforms, concatenation, and persistence, but do not
compute or store Jacobians/eigenvalues automatically. Use Linearization for
that analysis. The finder currently disables requires_grad on model parameters
and does not restore their previous flags.

Flow Fields And Projection
--------------------------

FlowFieldFinder fits two-component PCA from fit_states, or from the first
flow call when fitting data is omitted. PCA needs at least two samples and
two features. Both fit_states and flow states can use LSTM tuples, but direct
``transform`` and ``_reduce_traj`` calls require packed tensors.

Explicit axes have shape [2, D]. FlowFieldVisualizer can take an axis bank
[N_axes, D], with at least two rows. For LSTMs D = 2H, so the projection
includes both h and c. The inverse grid reconstructs the full packed state.
A hidden-only display requires an explicit choice of cell state at every grid
point; a standalone hidden trajectory cannot define the LSTM dynamics.

The nonlinear method advances reconstructed grid states one step. The linear
method uses an affine approximation around each supplied reference state and
also takes delta_inp matching inp. Inputs must have one sample per flattened
state; a shared input is not broadcast by the flow methods. ``follow_traj=True``
centers a grid at each projected state; otherwise the configured center is fixed.

Each FlowField contains grid [P, P, 2], x/y displacements [P, P], and speeds
[P, P], with P = num_points. Speeds are projected displacement magnitudes
normalized by their field maximum. They are not full-state speeds or energies;
the current normalization is undefined for an entirely stationary field.
A two-dimensional projection can hide unstable directions.

Interactive Visualization
-------------------------

Pass packed trajectories to ``FlowFieldVisualizer.visualize(inputs, states)``.
For LSTMs use width 2H and axes/fit_states in the same coordinates. The viewer
supports linear/nonlinear fields, paging, panning, zooming, trajectory overlays,
heatmaps, and contours. Linear input perturbations default to zeros.

When an axis bank is provided, Options > Axes stages the active selection.
The field regenerates when exactly two axes are selected. ``axis_labels``
names the rows; defaults are axis_1, axis_2, and so on.
