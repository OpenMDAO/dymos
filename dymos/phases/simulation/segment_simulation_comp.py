"""
SimulationPhase is an instance that resembles a Phase in structure but is intended for
use with scipy.solve_ivp to verify the accuracy of the implicit solutions of Dymos.
"""
from __future__ import print_function, division, absolute_import

import numpy as np

from six import iteritems

from scipy.integrate import solve_ivp

from openmdao.api import ExplicitComponent, OptionsDictionary

from ...utils.interpolate import LagrangeBarycentricInterpolant
from .ode_integration_interface import ODEIntegrationInterface




class SegmentSimulationComp(ExplicitComponent):
    """
    SegmentSimulationComp is a component which, given values for time, states, and controls
    within a given segment, explicitly simulates the segment using scipy.integrate.solve_ivp.

    The resulting states are captured at all nodes within the segment.
    """

    def initialize(self):
        self.options.declare('index', desc='the index of this segment in the parent phase.')
        self.options.declare('grid_data', desc='the grid data of the corresponding phase.')
        self.options.declare('ode_class',
                             desc='System defining the ODE')
        self.options.declare('ode_init_kwargs', types=dict, default={},
                             desc='Keyword arguments provided when initializing the ODE System')
        self.options.declare('time_options', types=OptionsDictionary)
        self.options.declare('state_options', types=dict)
        self.options.declare('control_options', types=dict)
        self.options.declare('design_parameter_options', types=dict)
        self.options.declare('input_parameter_options', types=dict)

    def setup(self):
        idx = self.options['index']
        gd = self.options['grid_data']
        nnps = gd.subset_num_nodes_per_segment['all'][idx]
        ncdsps = gd.subset_num_nodes_per_segment['control_disc'][idx]
        seg_idxs = gd.segment_indices[idx, :]

        self.ode_integration_interface = ODEIntegrationInterface(
            phase_name='',
            ode_class=self.options['ode_class'],
            time_options=self.options['time_options'],
            state_options=self.options['state_options'],
            control_options=self.options['control_options'],
            design_parameter_options=self.options['design_parameter_options'],
            input_parameter_options=self.options['input_parameter_options'],
            ode_init_kwargs=self.options['ode_init_kwargs'])

        self.add_input(name='time', val=np.ones(nnps),
                       units=self.options['time_options']['units'],
                       desc='Value of time at all nodes within the segment.')

        # self.add_input(name='time_phase', val=np.ones(nnps),
        #                units=self.options['time_options']['units'],
        #                desc='Value of phase elapsed time at all nodes within the segment.')

        # Setup the initial state vector for integration
        self.state_vec_size = 0
        for name, options in iteritems(self.options['state_options']):
            self.state_vec_size += np.prod(options['shape'])
            self.add_input(name='initial_states:{0}'.format(name), val=np.ones(options['shape']),
                           units=options['units'], desc='initial values of state {0}'.format(name))
            self.add_output(name='states:{0}'.format(name), val=np.ones((nnps,) + options['shape']),
                            units=options['units'],
                            desc='Values of state {0} at all nodes in the segment.'.format(name))
        self.initial_state_vec = np.zeros(self.state_vec_size)


        # Setup the control interpolants
        for name, options in iteritems(self.options['control_options']):
            self.add_input(name='controls:{0}'.format(name),
                           val=np.ones(((ncdsps,) + options['shape'])),
                           units=options['units'],
                           desc='Values of control {0} at control discretization '
                                'nodes within the .'.format(name))
            interp = LagrangeBarycentricInterpolant(gd.node_stau[seg_idxs[0]:seg_idxs[1]],
                                                    options['shape'])
            self.ode_integration_interface.control_interpolants[name] = interp

        for name, options in iteritems(self.options['design_parameter_options']):
            self.add_input(name='design_parameters:{0}'.format(name), val=np.ones(options['shape']),
                           units=options['units'],
                           desc='values of design parameter {0}.'.format(name))

        for name, options in iteritems(self.options['input_parameter_options']):
            self.add_input(name='input_parameters:{0}'.format(name), val=np.ones(options['shape']),
                           units=options['units'],
                           desc='values of input parameter {0}'.format(name))

        self.ode_integration_interface.prob.setup(check=False)

        # self.ode_integration_interface.setup(check=False)

        self.declare_partials(of='*', wrt='*', method='fd')


    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        gd = self.options['grid_data']
        nnps = gd.subset_num_nodes_per_segment['all'][self.options['index']]
        iface_prob = self.ode_integration_interface.prob

        # Create the vector of initial state values
        self.initial_state_vec[:] = 0.0
        pos = 0
        for name, options in iteritems(self.options['state_options']):
            size = np.prod(options['shape'])
            self.initial_state_vec[pos:pos + size] = \
                np.ravel(inputs['initial_states:{0}'.format(name)])
            pos += size

        # Setup the control interpolant
        for name, options in iteritems(self.options['control_options']):
            ctrl_vals = inputs['controls:{0}'.format(name)]
            self.ode_integration_interface.control_interpolants[name].setup(x0=inputs['time'][0],
                                                                            xf=inputs['time'][-1],
                                                                            f_j=ctrl_vals)

        # Set the values of the phase design parameters
        for param_name, options in iteritems(self.options['design_parameter_options']):
            val = inputs['design_parameters:{0}'.format(param_name)]
            iface_prob.set_val('design_parameters:{0}'.format(param_name),
                               value=val,
                               units=options['units'])

        # Set the values of the phase input parameters
        for param_name, options in iteritems(self.options['input_parameter_options']):
            iface_prob.set_val('input_parameters:{0}'.format(param_name),
                               value=inputs['input_parameters:{0}'.format(param_name)],
                               units=options['units'])

        # Perform the integration using solve_ivp
        sol = solve_ivp(fun=self.ode_integration_interface,
                        t_span=(inputs['time'][0], inputs['time'][-1]),
                        y0=self.initial_state_vec,
                        method='RK45',
                        t_eval=inputs['time'])

        # Extract the solution
        pos = 0
        for name, options in iteritems(self.options['state_options']):
            size = np.prod(options['shape'])
            outputs['states:{0}'.format(name)] = sol.y[pos:pos+size, :].T
            pos += size
