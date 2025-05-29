Dymos:  Open Source Optimization of Dynamic Multidisciplinary Systems
=====================================================================

[![Dymos Tests](https://github.com/OpenMDAO/dymos/actions/workflows/dymos_tests_workflow.yml/badge.svg)](https://github.com/OpenMDAO/dymos/actions/workflows/dymos_tests_workflow.yml) [![Coverage Status](https://coveralls.io/repos/github/OpenMDAO/dymos/badge.svg?branch=master&t=dJxu2Q)](https://coveralls.io/github/OpenMDAO/dymos?branch=master)

[![DOI](https://joss.theoj.org/papers/10.21105/joss.02809/status.svg)](https://doi.org/10.21105/joss.02809)



Dymos is a framework for the simulation and optimization of dynamical systems within the OpenMDAO Multidisciplinary Analysis and Optimization environment.
Dymos leverages implicit and explicit simulation techniques to simulate generic dynamic systems of arbitary complexity.

The software has two primary objectives:
-   Provide a generic ODE integration interface that allows for the analysis of dynamical systems.
-   Allow the user to solve optimal control problems involving dynamical multidisciplinary systems.

Installation
------------

The default installation of the developmental version of Dymos will install the minimum number of prerequisites:

```
python -m pip install dymos
```

More advanced installation instructions are available [here](https://openmdao.github.io/dymos/installation.html).

Citation
--------

See our [overview paper](https://joss.theoj.org/papers/10.21105/joss.02809) in the Journal of Open Source Software

If you use Dymos in your work, please cite:
```
@article{Falck2021,
  doi = {10.21105/joss.02809},
  url = {https://doi.org/10.21105/joss.02809},
  year = {2021},
  publisher = {The Open Journal},
  volume = {6},
  number = {59},
  pages = {2809},
  author = {Robert Falck and Justin S. Gray and Kaushik Ponnapalli and Ted Wright},
  title = {dymos: A Python package for optimal control of multidisciplinary systems},
  journal = {Journal of Open Source Software}
}
```



Documentation
-------------

Documentation for the current development version of Dymos is available at
[https://openmdao.github.io/dymos/](https://openmdao.github.io/dymos/) as well as on the OpenMDAO web site:
[https://openmdao.org/dymos/docs/latest/](https://openmdao.org/dymos/docs/latest/).
Archived versions for recent releases will also be found here:
[https://openmdao.org/dymos-documentation/](https://openmdao.org/dymos-documentation/)

Defining Ordinary Differential Equations
----------------------------------------

The first step in simulating or optimizing a dynamical system is to define the ordinary
differential equations to be integrated.  The user first builds an OpenMDAO model which has outputs
that provide the rates of the state variables.  This model can be an OpenMDAO model of arbitrary
complexity, including nested groups and components, layers of nonlinear solvers, etc.

Dymos solutions are constructed of one or more _Phases_.
When setting up a phase, we add state variables, dynamic controls, and parameters,
tell Dymos how the value of each should be connected to the ODE system, and tell Dymos
the variable paths in the system that contain the rates of our state variables that are to be
integrated.

Integrating Ordinary Differential Equations
-------------------------------------------

Dymos's solver-based pseudspectral transcriptions
provide the ability to numerically integrate the ODE system it is given.
Used in an optimal control context, these provide a shooting method in
which each iteration provides a physically viable trajectory.

Pseudospectral Methods
----------------------

Dymos currently supports the Radau Pseudospectral Method and high-order
Gauss-Lobatto transcriptions.  These implicit techniques rely on the
optimizer to impose "defect" constraints which enforce the physical
accuracy of the resulting trajectories.  To verify the physical
accuracy of the solutions, Dymos can explicitly integrate them using
variable-step methods.

Solving Optimal Control Problems
--------------------------------

Dymos uses the concept of _Phases_ to support optimal control of dynamical systems.
Users connect one or more Phases to construct trajectories.
Each Phase can have its own:

-   Optimal Control Transcription (Gauss-Lobatto or Radau Pseudospectral)
-   Equations of motion
-   Boundary and path constraints

Dymos Phases and Trajectories are ultimately just OpenMDAO Groups that can exist in
a problem along with numerous other models, allowing for the simultaneous
optimization of systems and dynamics.

```python
import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt

# First define a system which computes the equations of motion
class BrachistochroneEOM(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('v', val=np.zeros(nn), units='m/s', desc='velocity')
        self.add_input('theta', val=np.zeros(nn), units='rad', desc='angle of wire')
        self.add_output('xdot', val=np.zeros(nn), units='m/s', desc='x rate of change')
        self.add_output('ydot', val=np.zeros(nn), units='m/s', desc='y rate of change')
        self.add_output('vdot', val=np.zeros(nn), units='m/s**2', desc='v rate of change')

        # Ask OpenMDAO to compute the partial derivatives using complex-step
        # with a partial coloring algorithm for improved performance
        self.declare_partials(of='*', wrt='*', method='cs')
        self.declare_coloring(wrt='*', method='cs', show_summary=True)

    def compute(self, inputs, outputs):
        v, theta = inputs.values()
        outputs['vdot'] = 9.80665 * np.cos(theta)
        outputs['xdot'] = v * np.sin(theta)
        outputs['ydot'] = -v * np.cos(theta)

p = om.Problem()

# Define a Trajectory object
traj = p.model.add_subsystem('traj', dm.Trajectory())

# Define a Dymos Phase object with GaussLobatto Transcription
tx = dm.GaussLobatto(num_segments=10, order=3)
phase = dm.Phase(ode_class=BrachistochroneEOM, transcription=tx)

traj.add_phase(name='phase0', phase=phase)

# Set the time options
phase.set_time_options(fix_initial=True,
                       duration_bounds=(0.5, 10.0))
# Set the state options
phase.set_state_options('x', rate_source='xdot',
                        fix_initial=True, fix_final=True)
phase.set_state_options('y', rate_source='ydot',
                        fix_initial=True, fix_final=True)
phase.set_state_options('v', rate_source='vdot',
                        fix_initial=True, fix_final=False)
# Define theta as a control.
phase.add_control(name='theta', units='rad',
                  lower=0, upper=np.pi)
# Minimize final time.
phase.add_objective('time', loc='final')

# Set the driver.
p.driver = om.ScipyOptimizeDriver()

# Allow OpenMDAO to automatically determine total
# derivative sparsity pattern.
# This works in conjunction with partial derivative
# coloring to give a large speedup
p.driver.declare_coloring()

# Setup the problem
p.setup()

# Now that the OpenMDAO problem is setup, we can guess the
# values of time, states, and controls.
phase.set_time_val(initial=0.0, duration=2.0)

# States and controls here use a linearly interpolated
# initial guess along the trajectory.
phase.set_state_val('x', [0, 10], units='m')
phase.set_state_val('y', [10, 5], units='m')
phase.set_state_val('v', [0, 5], units='m/s')

# constant initial guess for control
phase.set_control_val('theta', 90, units='deg')

# Run the driver to solve the problem and generate default plots of
# state and control values vs time
dm.run_problem(p, make_plots=True, simulate=True)
```

When using the `make_plots=True` option above, the output directory generated within
the run directory will contain a file named `reports/traj_results_report.html` that
should look similar to this:

![Brachistochrone Solution](brachistochroneSolution.png "Brachistochrone Solution")
