# Segments of Phases

!!! info "Things you'll learn through this example"
    - What are segments?
    - How does the number and order of segments affect the solution?
    - How to use the Dymos run_problem function to find the right number of segments automatically.

## What are segments?

In the previous section we showed a converged trajectory that didn't really match the state propagation found using Scipy's variable step `solve_ivp` method.

{{ embed_test_plot('dymos.examples.oscillator.doc.test_doc_oscillator.TestDocOscillator.test_ivp_driver') }}

Why does this happen?
The implicit collocation techniques used by Dymos (the Radau Pseudospectral Method and Legendre-Gauss-Lobatto collocation) work by discretizing a continuous function (the state time-history) into a series of discrete points.
It does this by breaking the time domain of each phase into multiple polynomial _segments_.
On each segment, each state is treated as a continuous polynomial of some given order.
In Dymos, segments must have an order of **at least 3**.  That is also the default order for segments.

# How does the number and order of segments affect the solution?

Obviously, a single third-order polynomial won't be able to fit highly oscillator behavior.
In this case, our guess of using four segments (equally spaced) in the phase wasn't quite sufficient.
Lets try increasing that number to ten third-order segments.

{{ embed_test('dymos.examples.oscillator.doc.test_doc_oscillator.TestDocOscillator.test_ivp_driver_10_segs') }}

Alternatively, we could stick with 4 segments but give each a higher order (7 in this case).

{{ embed_test('dymos.examples.oscillator.doc.test_doc_oscillator.TestDocOscillator.test_ivp_driver_4_segs_7_order') }}

In both these cases, we obtained a better match of the Dynamics using either more segments or higher-order segments.
This gives the state interpolating polynomials enough freedom to more accurately match the true behavior of the system.
Increasing the number of segments and increasing the segment orders both increase the number of discrete points, and thus slow down the solution a bit.
Theres a balance to be found between using enough discretization points to get an accurate solution, and slowing down the analysis due to having an overabundance of points.
In general, using a high number of low-order segments is preferable to using fewer high-order segments because it makes the constraint jacobian more sparse.

In addition to the number and order of the segments, the user can also provide the transcription the argument `segment_ends`.
If `None`, the segments are equally distributed in time throughout the phase.
Otherwise, `semgent_ends` should be a monotonically increasing sequence of length `num_segments + 1`.

Each element in the sequence provides the location of a segment boundary in the phase.
The items in `segment_ends` are normalized by Dymos, so feel free to provide them in whatever scale makes sense.
That is, `semgent_ends=[0, 1, 2, 5]` is equivalent to `segent_ends=[10, 20, 30, 60]`.

## Letting Dymos automatically find the right segmentation of the phase

Manually tweaking the "grid" (the number of segments, their order, and relative spacing) isn't ideal.
In reality, another nested level of iteration is required:

1. `Problem.run_model()` evaluates the model and computes constraints and objectives based on the current design variables.
2. `Problem.run_driver()` iteratively calls `Problem.run_model()` while varying the design variables in order to find a feasible, optimal design point.
3. Some "outer" function iterates on `run_driver()`, varying the grid until a satisfactory accuracy is achieved.

This third level is filled by the role of automated grid refinement via the `dymos.run_problem`.  In the next section, we'll learn how to use automated grid refinement in Dymos.



