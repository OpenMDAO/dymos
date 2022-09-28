from collections import defaultdict
from fnmatch import filter
import warnings

import numpy as np
import openmdao.api as om
from openmdao.utils.om_warnings import issue_warning

from ..transcription_base import TranscriptionBase
from ...utils.misc import get_rate_units, _unspecified
from ...utils.introspection import configure_analytic_states_introspection, get_promoted_vars, get_targets, \
    get_source_metadata
from ...utils.indexing import get_src_indices_by_row
from ..grid_data import GridData
from .analytic_states_comp import AnalyticStatesComp
from .analytic_timeseries_output_comp import AnalyticTimeseriesOutputComp
from ..common.time_comp import TimeComp

class Analytic(TranscriptionBase):
    """
    Radau Pseudospectral Method Transcription.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.

    References
    ----------
    Garg, Divya et al. "Direct Trajectory Optimization and Costate Estimation of General Optimal
    Control Problems Using a Radau Pseudospectral Method." American Institute of Aeronautics
    and Astronautics, 2009.
    """
    def __init__(self, **kwargs):
        super(Analytic, self).__init__(**kwargs)
        self._rhs_source = 'rhs'

    def initialize(self):
        self.options.declare('grid')

    def init_grid(self):
        """
        Setup the GridData object for the Transcription.
        """
        self.grid_data = GridData(num_segments=self.options['num_segments'],
                                  transcription=self.options['grid'],
                                  transcription_order=self.options['order'],
                                  segment_ends=self.options['segment_ends'],
                                  compressed=self.options['compressed'])

    def setup_time(self, phase):
        """
        Setup the time component.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        grid_data = self.grid_data

        super().setup_time(phase)

        time_comp = TimeComp(num_nodes=grid_data.num_nodes, node_ptau=grid_data.node_ptau,
                             node_dptau_dstau=grid_data.node_dptau_dstau,
                             units=phase.time_options['units'],
                             initial_val=phase.time_options['initial_val'],
                             duration_val=phase.time_options['duration_val'])

        phase.add_subsystem('time', time_comp, promotes_inputs=['*'], promotes_outputs=['*'])

    def configure_time(self, phase):
        """
        Configure the inputs/outputs on the time component.

        This method assumes that target introspection has already been performed by the phase and thus
        options['targets'], options['time_phase_targets'], options['t_initial_targets'],
        and options['t_duration_targets'] are all correctly populated.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super(Analytic, self).configure_time(phase)
        phase.time.configure_io()

        options = phase.time_options
        ode = phase._get_subsystem(self._rhs_source)
        ode_inputs = get_promoted_vars(ode, iotypes='input')

        # The tuples here are (name, user_specified_targets, dynamic)
        for name, targets, dynamic in [('time', options['targets'], True),
                                       ('time_phase', options['time_phase_targets'], True)]:
            if targets:
                src_idxs = self.grid_data.subset_node_indices['all'] if dynamic else None
                phase.connect(name, [f'rhs.{t}' for t in targets], src_indices=src_idxs,
                              flat_src_indices=True if dynamic else None)

        for name, targets in [('t_initial', options['t_initial_targets']),
                              ('t_duration', options['t_duration_targets'])]:
            for t in targets:
                shape = ode_inputs[t]['shape']

                if shape == (1,):
                    src_idxs = None
                    flat_src_idxs = None
                    src_shape = None
                else:
                    src_idxs = np.zeros(self.grid_data.subset_num_nodes['all'])
                    flat_src_idxs = True
                    src_shape = (1,)

                phase.promotes('rhs', inputs=[(t, name)], src_indices=src_idxs,
                               flat_src_indices=flat_src_idxs, src_shape=src_shape)
            if targets:
                phase.set_input_defaults(name=name,
                                         val=np.ones((1,)),
                                         units=options['units'])

    def setup_controls(self, phase):
        """
        Setup the inputs/outputs for the controls.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def configure_controls(self, phase):
        """
        Configure the inputs/outputs for the controls.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def setup_polynomial_controls(self, phase):
        """
        Setup the inputs/outputs for the controls.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def configure_polynomial_controls(self, phase):
        """
        Configure the inputs/outputs for the polynomial controls.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def setup_states(self, phase):
        """
        Setup the states for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        grid_data = self.grid_data

        self.any_solved_segs = False
        self.any_connected_opt_segs = False

        phase.add_subsystem('states_comp', AnalyticStatesComp(),
                            promotes_inputs=['initial_states:*'], promotes_outputs=['initial_state_vals:*'])

    def configure_states_introspection(self, phase):
        """
        Configure state introspection to determine properties of states.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        ode = self._get_ode(phase)
        try:
            configure_analytic_states_introspection(phase.state_options, phase.time_options, phase.control_options,
                                                    phase.parameter_options, phase.polynomial_control_options, ode)
        except (ValueError, RuntimeError) as e:
            raise RuntimeError(f'Error during configure_states_introspection in phase {phase.pathname}.') from e

    def configure_states(self, phase):
        """
        Configure state connections post-introspection.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super().configure_states(phase)
        states_comp = phase._get_subsystem('states_comp')
        for state_name, options in phase.state_options.items():
            if options['fix_final']:
                raise ValueError('fix_final is not a valid option for states when using the '
                                 'Analytic transcription.')

            states_comp.add_state(state_name, options)

            if options['opt'] and not options['fix_initial']:
                phase.add_design_var(name=f'initial_states:{state_name}',
                                     lower=options['lower'],
                                     upper=options['upper'],
                                     scaler=options['scaler'],
                                     adder=options['adder'],
                                     ref0=options['ref0'],
                                     ref=options['ref'])

    def setup_ode(self, phase):
        """
        Setup the ode for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        ODEClass = phase.options['ode_class']
        grid_data = self.grid_data

        kwargs = phase.options['ode_init_kwargs']
        phase.add_subsystem('rhs',
                            subsys=ODEClass(num_nodes=grid_data.subset_num_nodes['all'],
                                            **kwargs))

    def configure_ode(self, phase):
        """
        Create connections to the introspected states.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        grid_data = self.grid_data
        ode_inputs = get_promoted_vars(phase.rhs, 'input')

        for name, options in phase.state_options.items():
            initial_targets = get_targets(ode_inputs, name=name, user_targets=options['initial_targets'])

            if initial_targets:
                phase.connect(f'initial_state_vals:{name}',
                              [f'rhs.{tgt}' for tgt in initial_targets])

    def setup_defects(self, phase):
        """
        Setup the defects for this transcription. The AnalyticTranscription has no defect constraints.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def setup_solvers(self, phase):
        """
        Setup the solvers for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def configure_solvers(self, phase):
        """
        Setup the solvers for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def setup_timeseries_outputs(self, phase):
        """
        Setup the timeseries for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        gd = self.grid_data

        for name, options in phase._timeseries.items():
            if options['transcription'] is None:
                ogd = None
            else:
                ogd = options['transcription'].grid_data

            timeseries_comp = AnalyticTimeseriesOutputComp(input_grid_data=gd,
                                                           output_grid_data=ogd,
                                                           output_subset=options['subset'],
                                                           time_units=phase.time_options['units'])

            phase.add_subsystem(name, subsys=timeseries_comp)

            phase.connect('dt_dstau', (f'{name}.dt_dstau'), flat_src_indices=True)

    def configure_defects(self, phase):
        """
        Configure defects, there are none in Analytic transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def _get_timeseries_var_source(self, var, output_name, phase):
        """
        Return the source path and indices for a given variable to be connected to a timeseries.

        Parameters
        ----------
        var : str
            Name of the timeseries variable whose source is desired.
        output_name : str
            Name of the timeseries output whose source is desired.
        phase : dymos.Phase
            Phase object containing the variable, either as state, time, control, etc., or as an ODE output.

        Returns
        -------
        meta : dict
            Metadata pertaining to the variable at the given path. This dict contains 'src' (the path to the
            timeseries source), 'src_idxs' (an array of the
            source indices), 'units' (the units of the source variable), and 'shape' (the shape of the variable at
            a given node).
        """
        gd = self.grid_data
        var_type = phase.classify_var(var)
        time_units = phase.time_options['units']

        transcription = phase.options['transcription']
        ode = transcription._get_ode(phase)
        ode_outputs = get_promoted_vars(ode, 'output')

        # The default for node_idxs, applies to everything except states and parameters.
        node_idxs = gd.subset_node_indices['all']

        meta = {}

        # Determine the path to the variable
        if var_type == 'time':
            path = 'time'
            src_units = time_units
            src_shape = (1,)
        elif var_type == 'time_phase':
            path = 'time_phase'
            src_units = time_units
            src_shape = (1,)
        elif var_type in ['indep_control', 'input_control']:
            path = f'control_values:{var}'
            src_units = phase.control_options[var]['units']
            src_shape = phase.control_options[var]['shape']
        elif var_type == 'control_rate':
            control_name = var[:-5]
            path = f'control_rates:{control_name}_rate'
            control_name = var[:-5]
            src_units = get_rate_units(phase.control_options[control_name]['units'], time_units, deriv=1)
            src_shape = phase.control_options[control_name]['shape']
        elif var_type == 'control_rate2':
            control_name = var[:-6]
            path = f'control_rates:{control_name}_rate2'
            src_units = get_rate_units(phase.control_options[control_name]['units'], time_units, deriv=2)
            src_shape = phase.control_options[control_name]['shape']
        elif var_type in ['indep_polynomial_control', 'input_polynomial_control']:
            path = f'polynomial_control_values:{var}'
            src_units = phase.polynomial_control_options[var]['units']
            src_shape = phase.polynomial_control_options[var]['shape']
        elif var_type == 'polynomial_control_rate':
            control_name = var[:-5]
            path = f'polynomial_control_rates:{control_name}_rate'
            control = phase.polynomial_control_options[control_name]
            src_units = get_rate_units(control['units'], time_units, deriv=1)
            src_shape = control['shape']
        elif var_type == 'polynomial_control_rate2':
            control_name = var[:-6]
            path = f'polynomial_control_rates:{control_name}_rate2'
            control = phase.polynomial_control_options[control_name]
            src_units = get_rate_units(control['units'], time_units, deriv=2)
            src_shape = control['shape']
        elif var_type == 'parameter':
            path = f'parameter_vals:{var}'
            # Timeseries are never a static_target
            node_idxs = np.zeros(gd.subset_num_nodes['all'], dtype=int)
            src_units = phase.parameter_options[var]['units']
            src_shape = phase.parameter_options[var]['shape']
        else:
            # Failed to find variable, assume it is in the ODE
            path = f'rhs.{var}'
            meta = get_source_metadata(ode_outputs, src=var)
            src_shape = meta['shape']
            src_units = meta['units']
            src_tags = meta['tags']
            if 'dymos.static_output' in src_tags:
                raise RuntimeError(f'ODE output {var} is tagged with "dymos.static_output" and cannot be a timeseries output.')

        src_idxs = om.slicer[node_idxs, ...]

        meta['src'] = path
        meta['src_idxs'] = src_idxs
        meta['units'] = src_units
        meta['shape'] = src_shape

        return meta

    def get_parameter_connections(self, name, phase):
        """
        Returns info about a parameter's target connections in the phase.

        Parameters
        ----------
        name : str
            Parameter name.
        phase : dymos.Phase
            The phase object to which this transcription instance applies.

        Returns
        -------
        list of (paths, indices)
            A list containing a tuple of target paths and corresponding src_indices to which the
            given design variable is to be connected.
        """
        connection_info = []

        if name in phase.parameter_options:
            options = phase.parameter_options[name]
            if not options['static_target']:
                src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)
                src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])
                if options['shape'] == (1,):
                    src_idxs = src_idxs.ravel()
            else:
                src_idxs_raw = np.zeros(1, dtype=int)
                src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])
                src_idxs = np.squeeze(src_idxs, axis=0)

            rhs_all_tgts = [f'rhs_all.{t}' for t in options['targets']]
            connection_info.append((rhs_all_tgts, (src_idxs,)))

        return connection_info

    def _requires_continuity_constraints(self, phase):
        """
        Tests whether state and/or control and/or control rate continuity are required.

        Parameters
        ----------
        phase : dymos.Phase
            The phase to which this transcription applies.

        Returns
        -------
        any_state_continuity : bool
            True if any state continuity is required to be enforced.
        any_control_continuity : bool
            True if any control value continuity is required to be enforced.
        any_control_rate_continuity : bool
            True if any control rate continuity is required to be enforced.
        """
        num_seg = self.grid_data.num_segments
        compressed = self.grid_data.compressed

        any_state_continuity = num_seg > 1 and not compressed
        any_control_continuity = any([opts['continuity'] for opts in phase.control_options.values()])
        any_control_continuity = any_control_continuity and num_seg > 1
        any_rate_continuity = any([opts['rate_continuity'] or opts['rate2_continuity']
                                   for opts in phase.control_options.values()])
        any_rate_continuity = any_rate_continuity and num_seg > 1

        return any_state_continuity, any_control_continuity, any_rate_continuity
