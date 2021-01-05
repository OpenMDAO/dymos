---
title: 'dymos: A Python package for optimal control of multidisciplinary systems'
tags:
  - Python
  - OpenMDAO
  - optimal control
  - trajectory optimization
  - multidisciplinary optimization
  - NASA
authors:
  - name: Rob Falck
    orcid: 0000-0001-9864-4928
    affiliation: 1
  - name: Justin Gray
    affiliation: 1
  - name: Kaushik Ponnapalli
    affiliation: 2
  - name: Ted Wright
    affiliation: 1
affiliations:
  - name: NASA Glenn Research Center
    index: 1
  - name: HX5 LLC
    index: 2
date: 26 June 2020
bibliography: paper.bib
---

# Summary

Dymos is a library for optimizing control schedules for dynamic systems --- sometimes referred to as  optimal control or trajectory optimization.
There are a number of other optimal control libraries that tackle similar kinds of problems, such as OTIS4 [@Paris2006], GPOPS-II [@Patterson2014GPOPSII],and CASADI [@Andersson2018].
These tools all rely on gradient based optimization to solve optimal control problems, though their methods of computing the gradients vary. 
Dymos is built on top of the OpenMDAO framework[@Gray2019a] and supports its modular derivative system which allows users to mix-and-match from finite-differencing, complex-step, hand-differentiated, and algorithmic differentiation. 
This flexibility allows Dymos to efficiently solve optimal control problems constructed with both ordinary differential equations (ODE) and differential algebraic equations (DAE). 

Dymos can also help solve more general optimization problems where dynamics are only one part in a larger system level model with additional --- potentially computationally expensive --- calculations that come before and after the dynamic calculations.  
These broader problems are commonly referred to co-design, controls-co-design, and multidisciplinary design optimization.
Dymos provides specific APIs and features that make it possible to integrate traditional optimal-control models into a co-design context, while still supporting analytic derivatives that are necessary for computational efficiency in these complex use cases. 
An example of a co-design problem that was solved with Dymos is the coupled trajectory-thermal design of an electric vertical takeoff and landing aircraft where the thermal management and propulsion systems were designed simultaneously with the flight trajectories to ensure no components overheated[@Hariton2020a].


# Difference between optimal-control and co-design
In the most general sense, both optimal-control and co-design problems are both are numerically valid ways of optimizing dynamic systems. 
There a vector of time varying state variables ($\bar{x}$) who's behavior is affected by time ($t$), a vector of dynamic controls ($\bar{u)}$), and a vector of static design parameters ($\bar{d}$). 
The evolution of the states over time is governed by an ordinary differential equation (ODE) or differential algebraic equation (DAE):
\begin{align}
  \dot{\bar{x}} = f_{ode}(\bar{x},t,\bar{u},\bar{d})
\end{align}


A general problem formulation will look like this:  

\begin{align*} 
\mathrm{Minimize}& \qquad \mathrm{J} = f_{obj}(\bar{x}\, t,\bar{u}, \bar{d}) \\
\mathrm{With Respect To:} & t, \bar{x}, \bar{u}, \bar{d} \\
\mathrm{Subject , to:}& \\
\mathrm{Dynamic , Constraints:}& \qquad \dot{\bar{x}} = f_{ode}(\bar{x}, t, \bar{u}, \bar{d}) \\
\mathrm{Time:}& \qquad {t}_{lb} \leq t \leq {t}{ub} \\
\mathrm{State , Variables:}& \qquad \bar{x}_{lb} \leq \bar{x} \leq \bar{x}{ub} \\
\mathrm{Dynamic , Controls:}& \qquad \bar{u}_{lb} \leq \bar{u} \leq \bar{u}{ub} \\ 
\mathrm{Design , Parameters:}& \qquad \bar{d}_{lb} \leq \bar{d} \leq \bar{d}{ub} \\ 
\mathrm{State Defect Constraints:}& \qquad g_\Delta(\bar{x}_0, t_0, \bar{u}, \bar{d}) = 0 \\
\mathrm{Path Constraints:}& \qquad  g_\mathrm{path}(\bar{x}_0, t_0, \bar{u}, \bar{d}) = 0 \\
\mathrm{Initial , Boundary , Constraints:}& \qquad \bar{g}_{0, lb} \leq g_{0}(\bar{x}_0, t_0, \bar{u}_0, \bar{d}) \leq \bar{g}_{0, ub} \\
\mathrm{Final , Boundary , Constraints:}& \qquad \bar{g}_{f, lb} \leq g_{f}(\bar{x}_f, t_f, \bar{u}_f, \bar{d}) \leq \bar{g}_{f, ub} \\ 
\end{align*}

In the mathematical sense what distinguishes optimal-control from co-design is the particulars of which design variables and constraints are actually considered. 
Pure optimal control problems deal with an already designed system and seek to maximize performance by adjusting dynamic quantities ($t, \bar{x}, \bar{u}$) such as position, speed, fuel-burned, battery state of charge, etc. 
Co-design problems simultaneously vary the design parameters of a system ($\bar{d}$) and its dynamic behavior ($t, \bar{x}, \bar{u}$) to reach maximum performance. 


In practice, the difference between optimal-control and co-design is not mathematical but instead more related to how the static and dymamic calculations are implemented and how complex each of them are. 
For very simple physical design parameters (e.g. the radius of a cannon ball, spring constants, linkage lengths, etc) it is common to integrate the design calculations directly into the ODE.
Even though the calculations are static in nature, they can easily be coded as part of the ODE and still fits well into the optimal-control paradigm. 
The optimization structure thus looks like this: 

![optimal control diagram](images/opt_control.png){width=45%}

However, not all calculations are can be handled in this way. 
When you need to split calculations up into a static component and a dynamic component, this would typically be called co-design. 
For example if the physical design problem included shaping of an airfoil using expensive numerical solutions to partial differential equations to predict drag, then you would not want to embed that PDE solver into the dynamic model. 
Instead you could set up a coupled model with the PDE solver going first, and passing a table of data to be interpolated to the dynamic model. 
Traditionally, this kind of co-design process would be done via sequential optimization with an manual outer design iteration between teams. 
One group would come up with a physical design, using their own internal optimization setup and then a second would take that and generate optimal control profiles for it. 
This kind of iterative sequential optimization look like this: 

![optimal control diagram](images/sequential_co_design.png){width=100%}

Dymos can support sequential co-design, but its unique value is that it also enables a more tightly coupled co-design process with a single top level optimizer handling both parts of the problem simultaneously. 
Coupled co-design is particularly challenging because it requires propagating derivative information from the static analysis to the dynamic analysis in an efficient way. 

![optimal control diagram](images/coupled_co_design.png){width=75%}


# ODE versus DAE

Optimal control software typically requires that the dynamics of the system be defined as a set of ordinary differential equations (ODE) that use explicit functions to compute the rates of the state variables to be time-integrated.
Sometimes the dynamics are instead posed as a set of differential algebraic equations (DAE), where some residual equations need to be satisfied implicitly in order to solve for the state rates. 
From the perspective of an optimal-control or co-design problem both ODE and DAE formulations provide state rates that need to be integrated over time. 
The difference is that ODEs are explicit functions which are relatively easy to differentiate, but DAEs are implicit functions which are much more difficult to differentiate. 
Since the derivatives are needed to perform optimization, DAEs are more challenging to optimize. 

One relatively common use case for DAEs is differential inclusions, in which the state time-history is posed as a dynamic control and the traditional control schedule needed to achieve that time-history is found using a nonlinear solver within the dynamic model [@Seywald1994].
For some problems differential inclusion provides a more natural and numerical beneficial design space for the optimizer to traverse,
but the nonlinear solver poses numerical challenges for computing derivatives for the optimizer.
A simple approach to this is to just finite-difference across the nonlinear solver, but this is has been shown to be expensive and numerically unstable[@gray2014derivatives]. 
Another option, taken by some optimal control libraries, is to apply monolithic algorithmic differentiation[@griewank2003mathematical] across the nonlinear solver.
While it does provide accurate derivatives, the monolithic approach is expensive and uses a lot of memory[@mader2008adjoint; @kenway2019effective]. 
The most efficient approach is to use a pair of analytic derivative approachs called the direct and adjoint methods, which were unified into a single unified derivative equation (UDE) by Hwang and Martins[@hwang2018b]. 

Dymos adopts the UDE approach which uses a linear solver to compute total derivatives needed by the optimizer using only partial derivatives of the residual equations in the DAE.
This approach offers two key advantages. 
First partial derivatives of the DAE residual equations are much less computationally challenging to compute. 
Second, using the OpenMDAO underpinnings of Dymos users can construct their DAE in a modular fashion and combine various methods of computing the partial derivatives via finite-difference, complex-step [@Martins2003CS], algorithmic differentiation, or hand differentiation as needed. 


## The Dymos perspective on optimal control

Dymos breaks the trajectory into chunks of time called _phases_.
Breaking the trajectory into phases provides several capabilities.
Intermediate constraints along a trajectory can be enforced by applying boundary constraint to a phase that begins or ends at the time of interest.
For instance, the optimal trajectory of a launch vehicle may be required to ascend vertically to clear a launch tower before it pitches over on its way to orbit.
Path constraints can be applied within each phase to bound some performance parameter within that phase.
For example, reentry vehicles may need to adjust their trajectory to limit aerodynamic heating.

Each phase in a trajectory can use its own separate ODE .
For instance, an aircraft with vertical takeoff and landing capability may use different ODEs for vertical flight and horizontal flight.
ODE's are implemented as standard OpenMDAO models which are passed to phases at instantiation time with some additional annotations to identify the states, state-rates, and control inputs.

Every phase uses its own specific time discretization tailored to the dynamics in that chunk of the time-history. 
If one part of a trajectory has fast dynamics and another has slow dynamics,
the time history can be broken into two phases with separate time discretizations.

In the optimal-control community, there are a number of different techniques for discretizing the time integration, each one is called a transcription. 
Some transcriptions are widely used, such as Euler or Runge-Kutta based transcriptions. 
While these common ones are used in some cases, most optimal-control practitioners favor a more specialized class of transcriptions called direct collocation --- based on a class of pseudospectral methods. 
Dymos supports two different collocation transcriptions: high-order Gauss-Lobatto [@Herman1996] and Radau [@Garg2009].
Both of these represent time-histories as piece-wise polynomials of at least 3rd order and are formulated in a way that makes it possible to efficiently compute the needed quantities to perform integration in a numerically rigorous fashion. 


In addition to choosing a transcription, each phase can be computed using an explicit or implicit form. 
Some caution must be taken here because the term "implicit" can be used to describe some time integration schemes (e.g. backwards Euler), but that is not what is meant in an optimal-control context. 
In optimal-control, an explicit phases is one where the full time history is computed starting from the initial value and propagating forwards or from the final value and propagating backwards. 
From the optimizers perspective it will set values for the design parameters ($\bar{d}$) and the controls ($\bar{u}$) and can expect to be given a physically valid time history as the output.
Wrapping an optimizer around an explicit phase gives what is traditionally called a "shooting method" in the optimal-control world.  
In contrast, implicit phases don't provide valid time histories on their own. 
Instead, they provide a set of defect constraints at specific discretization points which the optimizer must drive to 0 in order to converge the problem. 
In this case, the optimizer sees design parameters ($\bar{d}$), controls ($\bar{u}$), and the state time history ($\bar{x}$) as its design variables, and also gains the extra defect constraints ($g_d$) to keep the problem well posed. 
In the context of the multidisciplinary design optimization field, explicit phases are similar to the multidisciplinary design feasible (MDF) optimization architecture and implicit phases are similar to the simultaneous analysis and design (SAND) optimization architecture[@Martins2013]. 

Both implicit and explicit phases are useful in different circumstances. 
The explicit phases are more natural ways to formulate the problem to many because the match the way you would use time-integration without optimization.
However when used with optimization they are also more computationally expensive, 
sensitive to singularities in the ODE, 
and potentially unable to converge to a valid solution. 
Implicit phases tend to be less intuitive, since they don't provide valid time histories without a converged optimization. 
Their advantages are tend to be faster, more numerically stable, and more scalable --- though they are also highly sensitive to initial conditions and optimization scaling. 

Dymos supports both explicit and implicit phases for both its transcriptions, 
and even allows mixtures of implicit and explicit states within a phase. 
This flexibility is valuable because it allows users to tailor their optimization to suit their needs. 
Switching transcriptions and changing from implicit to explicit requires very minor code changes - typically a single line in the run-script.
As the given example shows, not every combination will work for a given problem but Dymos intentionally makes it easy to experiment with different combinations. 

# Choice of optimization algorithm 

Dymos does not ship with its own built in optimizer. 
It relies on whatever optimizers you have available in your OpenMDAO installation. 
OpenMDAO ships with an interface to the optimizers in SciPy [@2020SciPy-NMeth], 
and an additional wrapper for the pyoptsparse [@Wu_pyoptsparse_2020] library which has more powerful optimizer options such as SNOPT [@GilMS05] and IPOPT [@wachter2006].
OpenMDAO also allows users to integrate their own optimizer of choice, which Dymos can then seamlessly use with without any additional modifications.
For simple problems, Scipy's SLSQP optimizer generally works fine.
On more challenging optimal-control problems, higher quality optimizers are important for getting good performance.

## Statement of Need

When dealing with the design of complex systems that include transient behavior, co-design becomes critical[@garciasans2019].
Broadly there are two approaches: sequential co-design or coupled co-design [@allison2004complex].
The best choice depends on the degree of interaction, or coupling, between various sub-systems. 
If the coupling is strong a coupled co-design approach is necessary to achieve the best performance.

Though there are a number of effective optimal control libraries, they tend to assume that they are on top of the modeling stack.
They frame every optimization problem as if it was a pure optimal-control problem, and hence are best suited to be used in a sequential co-design style. 
This poses large challenges when expanding to tightly coupled problems, where the interactions between the static and dynamic systems are very strong. 

Dymos provides a set of unique capabilities that make coupled co-design possible via efficient gradient-based optimization methods.
It provides differentiated time-integration schemes that can generate transient models from user provided ODEs, 
along with APIs that enable users to couple these transient models with other models to form the co-design system while carrying the differentiation through that coupling.
It also supports efficient differentiation of DAE's that include implicit relationships, which allows for a much broader set of possible ways to pose transient models. 
These two features combined make Dymos capable of handling coupled co-design problems in a manner that is more efficient than a pure optimal-control approach. 


## Selected applications of Dymos

Dymos has been used to demonstrate the coupling of flight dynamics and subsystem thermal constraints in electrical aircraft applications [@Falck2017a; @Hariton2020a].
NASA's X-57 "Maxwell" is using Dymos for mission planning to maximize data collection while abiding the limits of battery storage capacity and subsystem temperatures [@Schnulo2018a; @Schnulo2019a].
Other authors have used Dymos to perform studies of aircraft acoustics [@Ingraham2020a] and the the design of supersonic aircraft with thermal fuel management systems [@Jasa2018a].

## Example usage of Dymos

As a simple use-case of Dymos, consider the classic brachistochrone optimal control problem.
In this problem, we seek the shape of a frictionless wire strung between two points of different heights such that a bead sliding along the wire moves from the first point to the second point in minimum time.
To find the solution in Dymos, we first define the ordinary differential equations that govern the motion of the bead.

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
        self.add_input('v', val=np.zeros(nn), units='m/s',
                       desc='velocity')

        self.add_input('theta', val=np.zeros(nn), units='rad',
                       desc='angle of wire')

        self.add_output('xdot', val=np.zeros(nn), units='m/s',
                        desc='velocity component in x')

        self.add_output('ydot', val=np.zeros(nn), units='m/s',
                        desc='velocity component in y')

        self.add_output('vdot', val=np.zeros(nn), units='m/s**2',
                        desc='acceleration magnitude')

        # Setup partials for the analytic derivatives
        # These all have diagonal partial-derivative jacobians
        ar = np.arange(self.options['num_nodes'])

        self.declare_partials(of='vdot', wrt='theta', rows=ar, cols=ar)
        self.declare_partials(of='xdot', wrt='*', rows=ar, cols=ar)
        self.declare_partials(of='ydot', wrt='*', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        v, theta = inputs.values()

        outputs['vdot'] = 9.80665 * np.cos(theta)
        outputs['xdot'] = v * np.sin(theta)
        outputs['ydot'] = -v * np.cos(theta)

    def compute_partials(self, inputs, jacobian):
        v, theta = inputs.values()

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        jacobian['vdot', 'theta'] = -9.80665 * sin_theta

        jacobian['xdot', 'v'] = sin_theta
        jacobian['xdot', 'theta'] = v * cos_theta

        jacobian['ydot', 'v'] = -cos_theta
        jacobian['ydot', 'theta'] = v * sin_theta
```

Having defined the ODE, we can now use Dymos to find the optimal time-history of the angle between the nadir and the wire (theta).

```python

p = om.Problem(model=om.Group())

# Define a Trajectory object
traj = dm.Trajectory()
p.model.add_subsystem('traj', subsys=traj)

# Define a Dymos Phase object with GaussLobatto Transcription
phase = dm.Phase(ode_class=BrachistochroneEOM,
                 transcription=dm.GaussLobatto(num_segments=10, order=3))
traj.add_phase(name='phase0', phase=phase)

# Set the time options
phase.set_time_options(fix_initial=True,
                       duration_bounds=(0.5, 10.0))

# Set the state options
phase.add_state('x', rate_source='xdot',
                fix_initial=True, fix_final=True)
phase.add_state('y', rate_source='ydot',
                fix_initial=True, fix_final=True)
phase.add_state('v', rate_source='vdot',
                fix_initial=True, fix_final=False)

# Define theta as a control.
phase.add_control(name='theta', units='rad',
                  lower=0, upper=np.pi)

# Minimize final time.
phase.add_objective('time', loc='final')

# Set the driver.
p.driver = om.ScipyOptimizeDriver()

# Allow OpenMDAO to automatically determine our sparsity pattern.
# Doing so can significant speed up the execution of dymos.
p.driver.declare_coloring()

# Setup the problem
p.setup()

# Now that the OpenMDAO problem is setup, we can guess the
# values of time, states, and controls.
p.set_val('traj.phase0.t_duration', 2.0)

# States and controls here use a linearly interpolated
# initial guess along the trajectory.
p.set_val('traj.phase0.states:x',
          phase.interpolate(ys=[0, 10], nodes='state_input'),
          units='m')

p.set_val('traj.phase0.states:y',
          phase.interpolate(ys=[10, 5], nodes='state_input'),
          units='m')

p.set_val('traj.phase0.states:v',
          phase.interpolate(ys=[0, 5], nodes='state_input'),
          units='m/s')

p.set_val('traj.phase0.controls:theta',
          phase.interpolate(ys=[90, 90], nodes='control_input'),
          units='deg')

# Run the driver to solve the problem
p.run_driver()

# Check the validity of our results by using
# scipy.integrate.solve_ivp to integrate the solution.
sim_out = traj.simulate()
```

Plotting the resulting state and controls gives the following:

![Brachistochrone Solution](brach_plots.png)

# Acknowledgements

Dymos was developed with funding from NASA's Transformational Tools and Technologies ($T^3$) Project.

# References
