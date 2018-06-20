Dymos:  Open Source Optimization of Dynamic Multidisciplinary Systems
=====================================================================

[![Build Status](https://travis-ci.com/OpenMDAO/dymos.svg?token=tUBGTjUY1qBbh4Htx3Sr&branch=master)](https://travis-ci.com/OpenMDAO/dymos) [![Coverage Status](https://coveralls.io/repos/github/OpenMDAO/dymos/badge.svg?branch=master&t=dJxu2Q)](https://coveralls.io/github/OpenMDAO/dymos?branch=master)


Dymos is a framework for the simulation and optimization of dynamical systems within the OpenMDAO Multidisciplinary Analysis and Optimization environment.
Dymos leverages implicit and explicit simulation techniques to simulate generic dynamic systems of arbitary complexity.  

The software has two primary objectives:
- Provide a generic ODE integration interface that allows for the analysis of dynamical systems.
- Allow the user to solve optimal control problems involving dynamical multidisciplinary systems.

Installation
------------

```
pip install git+https://github.com/OpenMDAO/dymos.git
```

Defining Ordinary Differential Equations
----------------------------------------

The first step in simulating or optimizing a dynamical system is to define the ordinary
differential equations to be integrated.  The user first builds an OpenMDAO model which has outputs
that provide the rates of the state variables.  This model can be an OpenMDAO model of arbitrary
complexity, including nested groups and components, layers of nonlinear solvers, etc.

Next we wrap our system with decorators that provide information regarding the states to be
integrated, which sources in the model provide their rates, and where any externally provided
parameters should be connected.  When used in an optimal control context, these external parameters
may serve as controls.

    import numpy as np
    from openmdao.api import ExplicitComponent
    
    from dymos import declare_time, declare_state, declare_parameter
    
    @declare_time(units='s')
    @declare_state('x', rate_source='xdot', units='m')
    @declare_state('y', rate_source='ydot', units='m')
    @declare_state('v', rate_source='vdot', targets=['v'], units='m/s')
    @declare_parameter('theta', targets=['theta'])
    @declare_parameter('g', units='m/s**2', targets=['g'])
    class BrachistochroneEOM(ExplicitComponent):
    
        def initialize(self):
            self.metadata.declare('num_nodes', types=int)
    
        def setup(self):
            nn = self.metadata['num_nodes']
    
            # Inputs
            self.add_input('v',
                           val=np.zeros(nn),
                           desc='velocity',
                           units='m/s')
    
            self.add_input('g',
                           val=9.80665*np.ones(nn),
                           desc='gravitational acceleration',
                           units='m/s/s')
    
            self.add_input('theta',
                           val=np.zeros(nn),
                           desc='angle of wire',
                           units='rad')
    
            self.add_output('xdot',
                            val=np.zeros(nn),
                            desc='velocity component in x',
                            units='m/s')
    
            self.add_output('ydot',
                            val=np.zeros(nn),
                            desc='velocity component in y',
                            units='m/s')
    
            self.add_output('vdot',
                            val=np.zeros(nn),
                            desc='acceleration magnitude',
                            units='m/s**2')
    
            self.add_output('check',
                            val=np.zeros(nn),
                            desc='A check on the solution: v/sin(theta) = constant',
                            units='m/s')
    
            # Setup partials
            arange = np.arange(self.metadata['num_nodes'])
    
            self.declare_partials(of='vdot', wrt='g', rows=arange, cols=arange, val=1.0)
            self.declare_partials(of='vdot', wrt='theta', rows=arange, cols=arange, val=1.0)
    
            self.declare_partials(of='xdot', wrt='v', rows=arange, cols=arange, val=1.0)
            self.declare_partials(of='xdot', wrt='theta', rows=arange, cols=arange, val=1.0)
    
            self.declare_partials(of='ydot', wrt='v', rows=arange, cols=arange, val=1.0)
            self.declare_partials(of='ydot', wrt='theta', rows=arange, cols=arange, val=1.0)
    
            self.declare_partials(of='check', wrt='v', rows=arange, cols=arange, val=1.0)
            self.declare_partials(of='check', wrt='theta', rows=arange, cols=arange, val=1.0)
    
        def compute(self, inputs, outputs):
            theta = inputs['theta']
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            g = inputs['g']
            v = inputs['v']
    
            outputs['vdot'] = g*cos_theta
            outputs['xdot'] = v*sin_theta
            outputs['ydot'] = -v*cos_theta
            outputs['check'] = v/sin_theta
    
        def compute_partials(self, inputs, jacobian):
            theta = inputs['theta']
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            g = inputs['g']
            v = inputs['v']
    
            jacobian['vdot', 'g'] = cos_theta
            jacobian['vdot', 'theta'] = -g*sin_theta
    
            jacobian['xdot', 'v'] = sin_theta
            jacobian['xdot', 'theta'] = v*cos_theta
    
            jacobian['ydot', 'v'] = -cos_theta
            jacobian['ydot', 'theta'] = v*sin_theta
    
            jacobian['check', 'v'] = 1/sin_theta
            jacobian['check', 'theta'] = -v*cos_theta/sin_theta**2
 

Integrating Ordinary Differential Equations
-------------------------------------------

dymos uses *Generalized Linear Methods* (GLM) to enable a variety of integration schemes to be applied to dynamical systems.
dymos's `ScipyODEIntegrator` provides an OpenMDAO group which simulates the ODE system it is given.

Solving Optimal Control Problems
--------------------------------

dymos uses the concept of *phases* to support optimal control of dynamical systems.
Users connect one or more phases to construct trajectories.
Each phase can have its own:

- Optimal Control Transcription (Gauss-Lobatto, Radau Pseudospectral, or GLM)
- Equations of motion
- Boundary and path constraints

As with `ScipyODEIntegrator`, each dymos `Phase` is ultimately just an OpenMDAO Group that can exist in
a problem along with numerous other groups.
