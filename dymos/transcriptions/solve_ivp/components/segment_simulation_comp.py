"""
SimulationPhase is an instance that resembles a Phase in structure but is intended for
use with scipy.solve_ivp to verify the accuracy of the implicit solutions of Dymos.
"""
import inspect
import numpy as np

from scipy.integrate import solve_ivp

import openmdao.api as om
from ....utils.interpolate import LagrangeBarycentricInterpolant
from ....utils.lgl import lgl
from .ode_integration_interface import ODEIntegrationInterface
from ....phase.options import TimeOptionsDictionary


class SegmentSimulationComp(om.ExplicitComponent):
    """
    Class definition for SegmentSimulationComp.

    SegmentSimulationComp is a component which, given values for time, states, and controls
    within a given segment, explicitly simulates the segment using scipy.integrate.solve_ivp.

    The resulting states are captured at all nodes within the segment.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def initialize(self):
        """
        Declare component options.
        """
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

        self.options.declare('parameter_options', default=None, types=dict, allow_none=True,
                             desc='Dictionary of parameter names/options for the segments '
                                  'parent Phase.')

        self.options.declare('ode_integration_interface', default=None, allow_none=True,
                             types=ODEIntegrationInterface,
                             desc='The instance of the ODE integration interface used to provide '
                                  'the ODE to scipy.integrate.solve_ivp in the segment.  If None,'
                                  ' a new one will be instantiated for this segment.')

        self.options.declare('output_nodes_per_seg', default=None, types=(int,), allow_none=True,
                             desc='If None, results are provided at the all nodes within each'
                                  'segment.  If an int (n) then results are provided at n '
                                  'equally distributed points in time within each segment.')

        self.recording_options['options_excludes'] = ['ode_integration_interface']

    def configure_io(self):
        """
        I/O creation is delayed until configure so we can determine variable shape and units.
        """
        idx = self.options['index']
        gd = self.options['grid_data']

        if self.options['output_nodes_per_seg'] is None:
            nnps_i = gd.subset_num_nodes_per_segment['all'][idx]
        else:
            nnps_i = self.options['output_nodes_per_seg']

        # Number of control discretization nodes per segment
        ncdsps = gd.subset_num_nodes_per_segment['control_disc'][idx]

        # Indices of the control disc nodes belonging to the current segment
        control_disc_seg_idxs = gd.subset_segment_indices['control_disc'][idx]

        # Segment tau values for the control disc nodes in the phase
        control_disc_stau = gd.node_stau[gd.subset_node_indices['control_disc']]

        # Segment tau values for the control disc nodes in the current segment
        control_disc_seg_stau = control_disc_stau[control_disc_seg_idxs[0]:control_disc_seg_idxs[1]]

        if self.options['ode_integration_interface'] is None:
            self.options['ode_integration_interface'] = ODEIntegrationInterface(
                ode_class=self.options['ode_class'],
                time_options=self.options['time_options'],
                state_options=self.options['state_options'],
                control_options=self.options['control_options'],
                polynomial_control_options=self.options['polynomial_control_options'],
                parameter_options=self.options['parameter_options'],
                ode_init_kwargs=self.options['ode_init_kwargs'])

        self.add_input(name='time', val=np.ones(nnps_i),
                       units=self.options['time_options']['units'],
                       desc='Time at all nodes within the segment.')

        self.add_input(name='time_phase', val=np.ones(nnps_i),
                       units=self.options['time_options']['units'],
                       desc='Phase elapsed time at all nodes within the segment.')

        self.add_input(name='t_initial', val=0.0, units=self.options['time_options']['units'],
                       desc='Initial time value in the phase.')

        self.add_input(name='t_duration', val=1.0, units=self.options['time_options']['units'],
                       desc='Total time duration of the phase.')

        # Setup the initial state vector for integration
        self.state_vec_size = 0
        for name, options in self.options['state_options'].items():
            self.state_vec_size += np.prod(options['shape'])
            self.add_input(name='initial_states:{0}'.format(name), val=np.ones((1,) + options['shape']),
                           units=options['units'], desc='initial values of state {0} '
                                                        'in the segment'.format(name))
            self.add_output(name='states:{0}'.format(name),
                            val=np.ones((nnps_i,) + options['shape']),
                            units=options['units'],
                            desc='Values of state {0} at all nodes in the segment.'.format(name))

        self.initial_state_vec = np.zeros(self.state_vec_size)

        self.options['ode_integration_interface'].prob.setup(check=False)

        # Setup the control interpolants
        if self.options['control_options']:
            for name, options in self.options['control_options'].items():
                self.add_input(name='controls:{0}'.format(name),
                               val=np.ones(((ncdsps,) + options['shape'])),
                               units=options['units'],
                               desc='Values of control {0} at control discretization '
                                    'nodes within the segment.'.format(name))
                interp = LagrangeBarycentricInterpolant(control_disc_seg_stau, options['shape'])
                self.options['ode_integration_interface'].set_interpolant(name, interp)

        if self.options['polynomial_control_options']:
            for name, options in self.options['polynomial_control_options'].items():
                poly_control_disc_ptau, _ = lgl(options['order'] + 1)
                self.add_input(name='polynomial_controls:{0}'.format(name),
                               val=np.ones(((options['order'] + 1,) + options['shape'])),
                               units=options['units'],
                               desc='Values of polynomial control {0} at control discretization '
                                    'nodes within the phase.'.format(name))
                interp = LagrangeBarycentricInterpolant(poly_control_disc_ptau, options['shape'])
                self.options['ode_integration_interface'].set_interpolant(name, interp)

        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        """
        Compute component outputs.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        """
        idx = self.options['index']
        gd = self.options['grid_data']
        iface_prob = self.options['ode_integration_interface'].prob

        # Create the vector of initial state values
        self.initial_state_vec[:] = 0.0
        pos = 0
        for name, options in self.options['state_options'].items():
            size = np.prod(options['shape'])
            self.initial_state_vec[pos:pos + size] = \
                np.ravel(inputs['initial_states:{0}'.format(name)])
            pos += size

        # Setup the control interpolants
        if self.options['control_options']:
            t0_seg = inputs['time'][0]
            tf_seg = inputs['time'][-1]
            for name, options in self.options['control_options'].items():
                ctrl_vals = inputs['controls:{0}'.format(name)]
                self.options['ode_integration_interface'].setup_interpolant(name,
                                                                            x0=t0_seg,
                                                                            xf=tf_seg,
                                                                            f_j=ctrl_vals)

        # Setup the polynomial control interpolants
        if self.options['polynomial_control_options']:
            t0_phase = inputs['t_initial']
            tf_phase = inputs['t_initial'] + inputs['t_duration']
            for name, options in self.options['polynomial_control_options'].items():
                ctrl_vals = inputs['polynomial_controls:{0}'.format(name)]
                self.options['ode_integration_interface'].setup_interpolant(name,
                                                                            x0=t0_phase,
                                                                            xf=tf_phase,
                                                                            f_j=ctrl_vals)

        # Set the values of t_initial and t_duration
        iface_prob.set_val('t_initial',
                           value=inputs['t_initial'],
                           units=self.options['time_options']['units'])

        iface_prob.set_val('t_duration',
                           value=inputs['t_duration'],
                           units=self.options['time_options']['units'])

        # Set the values of the phase parameters
        if self.options['parameter_options']:
            for param_name, options in self.options['parameter_options'].items():
                val = inputs['parameters:{0}'.format(param_name)]
                iface_prob.set_val('parameters:{0}'.format(param_name),
                                   value=val,
                                   units=options['units'])

        # Setup the evaluation times.
        if self.options['output_nodes_per_seg'] is None:
            # Output nodes given as subset, convert segment tau of nodes to time
            i1, i2 = gd.subset_segment_indices['all'][idx, :]
            indices = gd.subset_node_indices['all'][i1:i2]
            nodes_eval = gd.node_stau[indices]  # evaluation nodes in segment tau space
            t_initial = inputs['time'][0]
            t_duration = inputs['time'][-1] - t_initial
            t_eval = t_initial + 0.5 * (nodes_eval + 1) * t_duration
        else:
            # Output nodes given as number, linspace them across the segment
            t_eval = np.linspace(inputs['time'][0], inputs['time'][-1],
                                 self.options['output_nodes_per_seg'])

        # Perform the integration using solve_ivp
        sol = solve_ivp(fun=self.options['ode_integration_interface'],
                        t_span=(inputs['time'][0], inputs['time'][-1]),
                        y0=self.initial_state_vec,
                        method=self.options['method'],
                        atol=self.options['atol'],
                        rtol=self.options['rtol'],
                        t_eval=t_eval)

        if not sol.success:
            raise om.AnalysisError(f'solve_ivp failed: {sol.message}')

        # Extract the solution
        pos = 0
        for name, options in self.options['state_options'].items():
            size = np.prod(options['shape'])
            outputs['states:{0}'.format(name)] = sol.y[pos:pos+size, :].T
            pos += size
