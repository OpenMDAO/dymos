# How can I more efficiently use finite-differenced components in the ODE?

Sometimes it's overly burdensome to get the analytic partials for a component.
In this case, OpenMDAO can use finite-differencing to approximate the partials of that component and then use those approximated partials when assembling the total derivatives.
However, if a dense sub-jacobian pattern is prescribed somewhere within the ODE, it will effect all dependent calculations and cause related total jacobians to have dense patterns.
In effect, a dense partial-derivative jacobian destroys the sparsity pattern of the problem.
To alleviate this, OpenMDAO provides a mechanism to [color the partials of a single component](http://openmdao.org/twodocs/versions/latest/features/experimental/simul_coloring_fd_cs.html).

As an example, consider the minimum time-to-climb problem.
The ODE of this problem consists of several components.
In this case, we're going to switch one of these components from using analytic derivatives to a finite-difference approximation.
Here he use an option on the component so that we can toggle the use of partial coloring on and off for testing, but that's not generally necessary.

=== "class DynamicPressureCompFD"
{{ inline_source('dymos.examples.min_time_climb.doc.dynamic_pressure_comp_partial_coloring.DynamicPressureCompFD',
include_def=True,
include_docstring=True,
indent_level=0)
}}

!!! note
    When using finite-differenced partials, they should not be specified in the `compute_partials` method.
    In fact, if all partials in the component are being approximated, `compute_partials` should just be omitted.

In this usage of `declare_coloring`, we use the following arguments:

- `wrt=['*']`
This is used to specify that we wish to find sparse partials **w**ith **r**espect **t**o all inputs.
- `method=['fd']`
We're using finite differencing to approximate the partials for coloring.
This is separate from the method used to actually compute the units.
Using `'cs'` here (complex-step) will result in more accurate derivatives if model supports the use of complex inputs.
- `tol=1.0E-6`
Any value in the Jacobian with a value greater than this will be considered a non-zero.
Since finite differencing is used and it generally encounters noise on the order of 1.0E-8, this tolerance should be larger than that.
If using complex-step for the approximation method this tolerance can be smaller - as small as about 1.0E-15.
- `num_full_jacs`
Compute the full jacobian twice before determining the partial sparsity pattern.
- `min_improve_pct`
If the number of solves required to compute the derivatives isn't reduced by at least this amount, then coloring is ignored and the dense jacobian is used.
- `show_summary = True`
Print the sparsity of the partial derivative jacobian.  This will display something like:

```
Jacobian shape: (60, 120)  ( 1.67% nonzero)
FWD solves: 2   REV solves: 0
Total colors vs. total size: 2 vs 120  (98.3% improvement)

Sparsity computed using tolerance: 1e-06
Time to compute sparsity: 0.011868 sec.
Time to compute coloring: 0.001385 sec.
```

- `show_sparsity=True`
Display the sparsity pattern in standard output to provide a visual indication whether or not it is working.
Here, this outputs the jacobian of `rho` with two diagonal bands - one for each of the two inputs.

```
Approx coloring for 'traj.phases.phase0.rhs_col.aero.q_comp' (class DynamicPressureCompFD)
f.............................f............................. 0  q
.f.............................f............................ 1  q
..f.............................f........................... 2  q
...f.............................f.......................... 3  q
....f.............................f......................... 4  q
.....f.............................f........................ 5  q
......f.............................f....................... 6  q
.......f.............................f...................... 7  q
........f.............................f..................... 8  q
.........f.............................f.................... 9  q
..........f.............................f................... 10  q
...........f.............................f.................. 11  q
............f.............................f................. 12  q
.............f.............................f................ 13  q
..............f.............................f............... 14  q
...............f.............................f.............. 15  q
................f.............................f............. 16  q
.................f.............................f............ 17  q
..................f.............................f........... 18  q
...................f.............................f.......... 19  q
....................f.............................f......... 20  q
.....................f.............................f........ 21  q
......................f.............................f....... 22  q
.......................f.............................f...... 23  q
........................f.............................f..... 24  q
.........................f.............................f.... 25  q
..........................f.............................f... 26  q
...........................f.............................f.. 27  q
............................f.............................f. 28  q
.............................f.............................f 29  q
|rho
                              |v
```

The sparsity patterns of the resulting total-derivative jacobian matrices are shown below.
Finite differencing without partial derivative coloring causes the sparsity pattern to be dense for a large portion of the matrix.
Since the dynamic pressure affects all of the defect constraints, the algorithm treats each defect constraint as if it is _potentially_ dependent upon all altitude and velocity values throughout the phase.
However, if partial derivative coloring is used, OpenMDAO recovers the same sparsity pattern as the analytic derivative case.

{{ embed_test('dymos.examples.min_time_climb.doc.test_doc_min_time_climb_fd.TestMinTimeClimbForDocs.test_min_time_climb_for_docs_partial_coloring',
              plot_names=('Analytic Sparse Partials', 'Finite Difference Only', 'Finite Difference + Partial Coloring'),
              show_script=False,
              show_output=False) }}

### Performance comparison

In this instance, the following performance was noted for the minimum time-to-climb case with 30 Gauss-Lobatto segments.
Using OpenMDAO's partial derivative coloring buys back a signficant amount of performance lost to finite differencing.
It should be noted that the IPOPT option `alpha_for_y` can have a signficant impact on performance here.
The default 'primal' step results in faster convergence for the sparse analytic case, but results in problematic convergence for the finite-differenced versions.
Switching the option using `p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'` results in a 'safer' step size and faster convergence of the finite-differenced versions, at the expense of some time in the analytic case.
Using that setting, the analytic optimization time was roughly 8.7 seconds compared to 7.5 seconds for the finite difference case with coloring.
Dense finite differencing in that case, was roughly twice as fast, at 15.5 seconds.

| Derivative Type                  | Optimization Time (s) |
|----------------------------------|-----------------------|
| Sparse Analytic                  | 6.3                   |
| Finite Difference (Dense)        | 29.9                  |
| Finite Difference (with Coloring)| 10.5                  |

