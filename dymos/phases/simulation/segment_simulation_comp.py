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

from ...utils.interpolate import LagrangeBarycentricInterpolant
from .ode_integration_interface import ODEIntegrationInterface
from ..options import TimeOptionsDictionary


class SegmentSimulationComp(ExplicitComponent):
    """
    SegmentSimulationComp is a component which, given values for time, states, and controls
    within a given segment, explicitly simulates the segment using scipy.integrate.solve_ivp.

    The resulting states are captured at all nodes within the segment.
    """
    def __init__(self, **kwargs):

        super(SegmentSimulationComp, self).__init__(**kwargs)

        self.time_options = TimeOptionsDictionary()
        self.state_options = {}
        self.control_options = {}
        self.design_parameter_options = {}
        self.input_parameter_options = {}
        self.traj_parameter_options = {}

    def initialize(self):
        self.options.declare('index', desc='the index of this segment in the parent phase.')
        self.options.declare('grid_data', desc='the grid data of the corresponding phase.')
        self.options.declare('ode_class',
                             desc='System defining the ODE')
        self.options.declare('ode_init_kwargs', types=dict, default={},
                             desc='Keyword arguments provided when initializing the ODE System')
        self.options.declare('t_eval', types=(Sequence, np.ndarray),
                             desc='the times at which outputs are requested in the segment')

    def setup(self):
        idx = self.options['index']
        gd = self.options['grid_data']
        ncdsps = gd.subset_num_nodes_per_segment['control_disc'][idx]
        # print(idx, ncdsps)
        # print(gd.subset_segment_indices)
        # Indices of the control disc nodes belonging to the current segment
        control_disc_seg_idxs = gd.subset_segment_indices['control_disc'][idx]
        # Segment tau values for the control disc nodes in the phase
        control_disc_stau = gd.node_stau[gd.subset_node_indices['control_disc']]
        # Segment tau values for the control disc nodes in the current segment
        control_disc_seg_stau = control_disc_stau[control_disc_seg_idxs[0]:control_disc_seg_idxs[1]]
        # print('segment ', idx)
        # print(gd.subset_node_indices['control_disc'])
        # print(gd.node_stau[gd.subset_node_indices['control_disc']])
        # print(control_disc_seg_idxs)
        # print(gd.node_stau[gd.subset_node_indices['control_disc']][control_disc_seg_idxs[0]:control_disc_seg_idxs[1]])
        # print(control_disc_stau[control_disc_seg_idxs[0]:control_disc_seg_idxs[1]])
        # print(control_disc_seg_stau)
        # print('end')
        # # print(control_disc_seg_idxs)
        # # print(control_disc_stau)
        # if idx == 3:
        #     exit(0)
        # print(control_disc_idxs_seg_i)
        # exit(0)
        num_output_points = len(self.options['t_eval'])
        # seg_idxs = gd.segment_indices[idx, :]

        self.ode_integration_interface = ODEIntegrationInterface(
            phase_name='',
            ode_class=self.options['ode_class'],
            time_options=self.time_options,
            state_options=self.state_options,
            control_options=self.control_options,
            design_parameter_options=self.design_parameter_options,
            input_parameter_options=self.input_parameter_options,
            traj_parameter_options=self.traj_parameter_options,
            ode_init_kwargs=self.options['ode_init_kwargs'])

        # Setup the initial state vector for integration
        self.state_vec_size = 0
        for name, options in iteritems(self.state_options):
            self.state_vec_size += np.prod(options['shape'])
            self.add_input(name='initial_states:{0}'.format(name), val=np.ones(options['shape']),
                           units=options['units'], desc='initial values of state {0}'.format(name))
            self.add_output(name='states:{0}'.format(name),
                            val=np.ones((num_output_points,) + options['shape']),
                            units=options['units'],
                            desc='Values of state {0} at t_eval.'.format(name))
        self.initial_state_vec = np.zeros(self.state_vec_size)

        # Setup the control interpolants
        for name, options in iteritems(self.control_options):
            self.add_input(name='controls:{0}'.format(name),
                           val=np.ones(((ncdsps,) + options['shape'])),
                           units=options['units'],
                           desc='Values of control {0} at control discretization '
                                'nodes within the .'.format(name))
            # print(gd.node_stau[control_disc_seg_idxs[0]:control_disc_seg_idxs[1]])
            interp = LagrangeBarycentricInterpolant(control_disc_seg_stau, options['shape'])
            self.ode_integration_interface.control_interpolants[name] = interp

        for name, options in iteritems(self.design_parameter_options):
            self.add_input(name='design_parameters:{0}'.format(name), val=np.ones(options['shape']),
                           units=options['units'],
                           desc='values of design parameter {0}.'.format(name))

        for name, options in iteritems(self.input_parameter_options):
            self.add_input(name='input_parameters:{0}'.format(name), val=np.ones(options['shape']),
                           units=options['units'],
                           desc='values of input parameter {0}'.format(name))

        for name, options in iteritems(self.traj_parameter_options):
            self.add_input(name='traj_parameters:{0}'.format(name), val=np.ones(options['shape']),
                           units=options['units'],
                           desc='values of trajectory parameter {0}'.format(name))

        self.ode_integration_interface.prob.setup(check=False)

        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        iface_prob = self.ode_integration_interface.prob
        t_eval = self.options['t_eval']

        # Create the vector of initial state values
        self.initial_state_vec[:] = 0.0
        pos = 0
        for name, options in iteritems(self.state_options):
            size = np.prod(options['shape'])
            self.initial_state_vec[pos:pos + size] = \
                np.ravel(inputs['initial_states:{0}'.format(name)])
            pos += size

        # Setup the control interpolant
        for name, options in iteritems(self.control_options):
            ctrl_vals = inputs['controls:{0}'.format(name)]
            self.ode_integration_interface.control_interpolants[name].setup(x0=t_eval[0],
                                                                            xf=t_eval[-1],
                                                                            f_j=ctrl_vals)

        # Set the values of the phase design parameters
        for param_name, options in iteritems(self.design_parameter_options):
            val = inputs['design_parameters:{0}'.format(param_name)]
            iface_prob.set_val('design_parameters:{0}'.format(param_name),
                               value=val,
                               units=options['units'])

        # Set the values of the phase input parameters
        for param_name, options in iteritems(self.input_parameter_options):
            iface_prob.set_val('input_parameters:{0}'.format(param_name),
                               value=inputs['input_parameters:{0}'.format(param_name)],
                               units=options['units'])

        # Set the values of the trajectory parameters
        for param_name, options in iteritems(self.traj_parameter_options):
            iface_prob.set_val('traj_parameters:{0}'.format(param_name),
                               value=inputs['traj_parameters:{0}'.format(param_name)],
                               units=options['units'])

        # Perform the integration using solve_ivp
        t_eval = self.options['t_eval']
        sol = solve_ivp(fun=self.ode_integration_interface,
                        t_span=(t_eval[0], t_eval[-1]),
                        y0=self.initial_state_vec,
                        method='RK45',
                        t_eval=t_eval)

        # Extract the solution
        pos = 0
        for name, options in iteritems(self.state_options):
            size = np.prod(options['shape'])
            outputs['states:{0}'.format(name)] = sol.y[pos:pos+size, :].T
            pos += size
