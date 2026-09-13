User Guide
==========

Preparing Trajectories
----------------------

Run a PyTorch RNN on representative inputs before analysis. Most RNNToolkit
methods accept one-dimensional samples or tensors with leading batch and time
dimensions; the analysis tools flatten leading dimensions into samples when
needed. Keep input samples aligned with their corresponding hidden states.

Linearization
-------------

``Linearization`` computes the Jacobians of one-step RNN dynamics with respect
to hidden state and input. Its eigendecomposition exposes local modes around a
chosen state. Supply one-dimensional input and hidden-state tensors to ``jacobian``.
``eigendecomposition`` accepts only the hidden state and evaluates its recurrent
Jacobian at zero input.

Fixed Points
------------

``FixedPointFinder`` optimizes initial hidden states toward states whose
one-step update changes as little as possible under a fixed external input.
Initial guesses should cover the state-space region of interest; sampling them
from trajectories usually gives better coverage than arbitrary points.

The finder returns both unique fixed points and the full collection produced by
all initializations. ``FixedPointCollection`` supports filtering, indexing,
coordinate transforms, concatenation, persistence, and access to optimization
and stability metadata.

Dimensionality Reduction
------------------------

Flow-field tools operate in a two-dimensional plane. They fit PCA from
``fit_states`` or use explicit axes. ``FlowFieldFinder`` accepts one ``[2, D]``
basis. ``FlowFieldVisualizer`` additionally accepts an ``[N, D]`` axis bank
with ``N >= 2`` and displays the first two axes initially. Use representative
fit states so the PCA plane captures the dynamics of interest. Explicit axes
are useful when comparing multiple runs in the same coordinate system.

Flow Fields
-----------

``FlowFieldFinder`` builds a grid in the reduced plane, lifts it into hidden
state space, and evaluates local motion. ``find_nonlinear_flow`` uses the full
RNN update. ``find_linear_flow`` uses a first-order approximation and therefore
also requires an input perturbation tensor aligned with inputs and states.

Set ``follow_traj=True`` to center a separate grid on every trajectory sample.
Otherwise, ``x_center`` and ``y_center`` define one shared grid.

Interactive Visualization
-------------------------

``FlowFieldVisualizer`` wraps flow-field computation in a Pygame interface. It
supports nonlinear and linear views, paging through samples, panning, zooming,
grid-density controls, trajectory overlays, heatmaps, and contours. When an
axis bank is provided, the Options > Axes side panel stages the active axes.
Any number may be toggled while editing; the flow is regenerated only when the
selection contains exactly two axes. Pass ``axis_labels`` to name its rows; otherwise they are named
``axis_1``, ``axis_2``, and so on. For linear flow, pass ``delta_inputs`` to
``visualize``; omitted perturbations default to zeros.
