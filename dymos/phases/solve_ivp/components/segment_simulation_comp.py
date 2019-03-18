"""
SimulationPhase is an instance that resembles a Phase in structure but is intended for
use with scipy.solve_ivp to verify the accuracy of the implicit solutions of Dymos.
"""
from __future__ import print_function, division, absolute_import

from collections import Sequence

import numpy as np

from six import iteritems

from scipy.integrate import solve_ivp

from openmdao.api import ExplicitComponent

from ....utils.interpolate import LagrangeBarycentricInterpolant
from .ode_integration_interface import ODEIntegrationInterface
from ...options import TimeOptionsDictionary


class SegmentSimulationComp(ExplicitComponent):
    """
    SegmentSimulationComp is a component which, given values for time, states, and controls
    within a given segment, explicitly simulates the segment using scipy.integrate.solve_ivp.

    The resulting states are captured at all nodes within the segment.
    """
    def __init__(self, **kwargs):

        super(SegmentSimulationComp, self).__init__(**kwargs)

    def initialize(self):
        self.options.declare('index', desc='the index of this segment in the parent phase.')

        self.options.declare('grid_data', desc='the grid data of the corresponding phase.')

        self.options.declare('method', default='RK45', values=('RK45', 'RK23', 'BDF', 'Radau'),
                             desc='The integrator used within scipy.integrate.solve_ivp. Currently '
                                  'supports \'RK45\', \'RK23\', and \'BDF\'.')

        self.options.declare('atol', default=1.0E-6, types=(float,),
                             desc='Absolute tolerance passed to scipy.integrate.solve_ivp.')

        self.options.declare('rtol', default=1.0E-6, types=(float,),
                             desc='Relative tolerance passed to scipy.integrate.solve_ivp.')

        self.options.declare('ode_class',
                             desc='System defining the ODE')
        self.options.declare('ode_init_kwargs', types=dict, default={},
                             desc='Keyword arguments provided when initializing the ODE System')

        self.options.declare('time_options', types=TimeOptionsDictionary,
                             desc='Time options for the phase')

        self.options.declare('state_options', types=dict,
                             desc='Dictionary of state names/options for the segments parent Phase')

        self.options.declare('control_options', default=None, types=dict, allow_none=True,
                             desc='Dictionary of control names/options for the segments parent Phase.')

        self.options.declare('polynomial_control_options', default=None, types=dict, allow_none=True,
                             desc='Dictionary of polynomial control names/options for the segments '
                                  'parent Phase.')

        self.options.declare('design_parameter_options', default=None, types=dict, allow_none=True,
                             desc='Dictionary of design parameter names/options for the segments '
                                  'parent Phase.')

        self.options.declare('input_parameter_options', default=None, types=dict, allow_none=True,
                             desc='Dictionary of input parameter names/options for the segments '
                                  'parent Phase.')

        self.options.declare('traj_parameter_options', default=None, types=dict, allow_none=True,
                             desc='Dictionary of traj parameter names/options for the segments '
                                  'parent Phase.')

        self.options.declare('ode_integration_interface', default=None, allow_none=True,
                             types=ODEIntegrationInterface,
                             desc='The instance of the ODE integration interface used to provide ' \
                                  'the ODE to scipy.integrate.solve_ivp in the segment.  If None,' \
                                  ' a new one will be instantiated for this segment.')


    def setup(self):
        idx = self.options['index']
        gd = self.options['grid_data']
        nnps_i = gd.subset_num_nodes_per_segment['all'][idx]

        # # Number of control discretization nodes per segment
        # ncdsps = gd.subset_num_nodes_per_segment['control_disc'][idx]
        #
        # # Indices of the control disc nodes belonging to the current segment
        # control_disc_seg_idxs = gd.subset_segment_indices['control_disc'][idx]
        #
        # # Segment tau values for the control disc nodes in the phase
        # control_disc_stau = gd.node_stau[gd.subset_node_indices['control_disc']]
        #
        # # Segment tau values for the control disc nodes in the current segment
        # control_disc_seg_stau = control_disc_stau[control_disc_seg_idxs[0]:control_disc_seg_idxs[1]]

        #num_output_points = len(self.options['t_eval'])

        if self.options['ode_integration_interface'] is None:
            self.options['ode_integration_interface'] = ODEIntegrationInterface(
                ode_class=self.options['ode_class'],
                time_options=self.options['time_options'],
                state_options=self.options['state_options'],
                control_options=self.options['control_options'],
                polynomial_control_options=self.options['polynomial_control_options'],
                design_parameter_options=self.options['design_parameter_options'],
                input_parameter_options=self.options['input_parameter_options'],
                traj_parameter_options=self.options['traj_parameter_options'],
                ode_init_kwargs=self.options['ode_init_kwargs'])

        self.add_input(name='time', val=np.ones(gd.subset_num_nodes_per_segment['all'][idx]),
                       units=self.options['time_options']['units'],
                       desc='Time at all nodes within the segment.')

        self.add_input(name='time_phase', val=np.ones(gd.subset_num_nodes_per_segment['all'][idx]),
                       units=self.options['time_options']['units'],
                       desc='Phase elapsed time at all nodes within the segment.')

        self.add_input(name='t_initial', val=0.0, units=self.options['time_options']['units'],
                       desc='Initial time value in the phase.')

        self.add_input(name='t_duration', val=1.0, units=self.options['time_options']['units'],
                       desc='Total time duration of the phase.')

        # Setup the initial state vector for integration
        self.state_vec_size = 0
        for name, options in iteritems(self.options['state_options']):
            self.state_vec_size += np.prod(options['shape'])
            self.add_input(name='initial_states:{0}'.format(name), val=np.ones((1,) + options['shape']),
                           units=options['units'], desc='initial values of state {0} '
                                                        'in the segment'.format(name))
            self.add_output(name='states:{0}'.format(name),
                            val=np.ones((nnps_i,) + options['shape']),
                            units=options['units'],
                            desc='Values of state {0} at all nodes in the segment.'.format(name))

        self.initial_state_vec = np.zeros(self.state_vec_size)

        # Setup the control interpolants
        if self.options['control_options']:
            for name, options in iteritems(self.options['control_options']):
                self.add_input(name='controls:{0}'.format(name),
                               val=np.ones(((ncdsps,) + options['shape'])),
                               units=options['units'],
                               desc='Values of control {0} at control discretization '
                                    'nodes within the .'.format(name))
                interp = LagrangeBarycentricInterpolant(control_disc_seg_stau, options['shape'])
                self.ode_integration_interface.control_interpolants[name] = interp

        if self.options['polynomial_control_options']:
            for name, options in iteritems(self.options['polynomial_control_options']):
                self.ode_integration_interface.polynomial_control_interpolants[name] = \
                    self.polynomial_control_interpolants[name]

        if self.options['design_parameter_options']:
            for name, options in iteritems(self.options['design_parameter_options']):
                self.add_input(name='design_parameters:{0}'.format(name), val=np.ones(options['shape']),
                               units=options['units'],
                               desc='values of design parameter {0}.'.format(name))

        if self.options['input_parameter_options']:
            for name, options in iteritems(self.options['input_parameter_options']):
                self.add_input(name='input_parameters:{0}'.format(name), val=np.ones(options['shape']),
                               units=options['units'],
                               desc='values of input parameter {0}'.format(name))

        if self.options['traj_parameter_options']:
            for name, options in iteritems(self.options['traj_parameter_options']):
                self.add_input(name='traj_parameters:{0}'.format(name), val=np.ones(options['shape']),
                               units=options['units'],
                               desc='values of trajectory parameter {0}'.format(name))

        self.options['ode_integration_interface'].prob.setup(check=False)

        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        iface_prob = self.options['ode_integration_interface'].prob
        # t_eval = self.options['t_eval']

        # Create the vector of initial state values
        self.initial_state_vec[:] = 0.0
        pos = 0
        for name, options in iteritems(self.options['state_options']):
            size = np.prod(options['shape'])
            self.initial_state_vec[pos:pos + size] = \
                np.ravel(inputs['initial_states:{0}'.format(name)])
            pos += size

        # Setup the control interpolants
        if self.options['control_options']:
            t0_seg, tf_seg = inputs['t_seg_ends']
            for name, options in iteritems(self.options['control_options']):
                ctrl_vals = inputs['controls:{0}'.format(name)]
                self.ode_integration_interface.control_interpolants[name].setup(x0=t0_seg,
                                                                                xf=tf_seg,
                                                                                f_j=ctrl_vals)

        # Setup the polynomial control interpolants
        if self.options['control_options']:
            t0_phase = inputs['t_initial']
            tf_phase = inputs['t_initial'] + inputs['t_duration']
            for name, options in iteritems(self.options['control_options']):
                ctrl_vals = inputs['polynomial_controls:{0}'.format(name)]
                self.ode_integration_interface.control_interpolants[name].setup(x0=t0_phase,
                                                                                xf=tf_phase,
                                                                                f_j=ctrl_vals)

        # Set the values of t_initial and t_duration
        iface_prob.set_val('t_initial',
                           value=inputs['t_initial'],
                           units=self.options['time_options']['units'])

        iface_prob.set_val('t_duration',
                           value=inputs['t_duration'],
                           units=self.options['time_options']['units'])

        # Set the values of the phase design parameters
        if self.options['design_parameter_options']:
            for param_name, options in iteritems(self.options['design_parameter_options']):
                val = inputs['design_parameters:{0}'.format(param_name)]
                iface_prob.set_val('design_parameters:{0}'.format(param_name),
                                   value=val,
                                   units=options['units'])

        # Set the values of the phase input parameters
        if self.options['input_parameter_options']:
            for param_name, options in iteritems(self.options['input_parameter_options']):
                iface_prob.set_val('input_parameters:{0}'.format(param_name),
                                   value=inputs['input_parameters:{0}'.format(param_name)],
                                   units=options['units'])

        # Set the values of the trajectory parameters
        if self.options['traj_parameter_options']:
            for param_name, options in iteritems(self.options['traj_parameter_options']):
                iface_prob.set_val('traj_parameters:{0}'.format(param_name),
                                   value=inputs['traj_parameters:{0}'.format(param_name)],
                                   units=options['units'])

        # Perform the integration using solve_ivp
        sol = solve_ivp(fun=self.options['ode_integration_interface'],
                        t_span=(inputs['time'][0], inputs['time'][-1]),
                        y0=self.initial_state_vec,
                        method=self.options['method'],
                        atol=self.options['atol'],
                        rtol=self.options['rtol'],
                        t_eval=inputs['time'])
                        # TODO: Eventually we need to return states at evaluation nodes
                        # t_eval=np.concatenate((inputs['t0_seg'], inputs['tf_seg'])))

        # Extract the solution
        pos = 0
        for name, options in iteritems(self.options['state_options']):
            size = np.prod(options['shape'])
            # TODO: Eventually we need to return states at evaluation nodes
            outputs['states:{0}'.format(name)] = sol.y[pos:pos+size, :].T
            pos += size
