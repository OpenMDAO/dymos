import numpy as np
import openmdao.api as om

from .birkhoff_collocation_comp import BirkhoffCollocationComp
from .birkhoff_state_resid_comp import BirkhoffStateResidComp

from ...grid_data import GridData
from ....phase.options import TimeOptionsDictionary


class BirkhoffIterGroup(om.Group):
    """
    Class definition for the BirkhoffIterGroup.

    This group allows for iteration of the state variables and initial _or_ final value of the state
    depending on the direction of the solve.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def initialize(self):
        """
        Declare group options.
        """
        self.options.declare('state_options', types=dict,
                             desc='Dictionary of options for the states.')
        self.options.declare('time_options', types=TimeOptionsDictionary,
                             desc='Options for time in the phase.')
        self.options.declare('grid_data', types=GridData, desc='Container object for grid info.')
        self.options.declare('ode_class', default=None,
                             desc='Callable that instantiates the ODE system.',
                             recordable=False)
        self.options.declare('ode_init_kwargs', types=dict, default={},
                             desc='Keyword arguments provided when initializing the ODE System')

    def setup(self):
        """
        Define the structure of the control group.
        """
        gd = self.options['grid_data']
        nn = gd.subset_num_nodes['all']
        state_options = self.options['state_options']
        time_options = self.options['time_options']
        ode_class = self.options['ode_class']
        ode_init_kwargs = self.options['ode_init_kwargs']

        self.add_subsystem('ode_all', subsys=ode_class(num_nodes=nn, **ode_init_kwargs))

        self.add_subsystem('collocation_comp',
                           subsys=BirkhoffCollocationComp(grid_data=gd,
                                                          state_options=state_options,
                                                          time_units=time_options['units']),
                           promotes_inputs=['*'], promotes_outputs=['*'])

        if any([opts['solve_segments'] in ('forward', 'backward') for opts in state_options.values()]):
            self.add_subsystem('states_balance_comp', subsys=BirkhoffStateResidComp(),
                               promotes_inputs=['*'], promotes_outputs=['*'])

    def _configure_desvars(self, name, options):
        state_name = f'states:{name}'
        initial_state_name = f'initial_states:{name}'
        final_state_name = f'final_states:{name}'
        state_rate_name = f'state_rates:{name}'

        solve_segs = options['solve_segments']
        opt = options['opt']

        ib = (None, None) if options['initial_bounds'] is None else options['initial_bounds']
        fb = (None, None) if options['final_bounds'] is None else options['final_bounds']
        lower = options['lower']
        upper = options['upper']
        scaler = options['scaler']
        adder = options['adder']
        ref0 = options['ref0']
        ref = options['ref']
        fix_initial = options['fix_initial']
        fix_final = options['fix_final']

        free_vars = {state_name, state_rate_name, initial_state_name, final_state_name}

        if solve_segs == 'forward':
            implicit_outputs = {state_name, state_rate_name, final_state_name}
        elif solve_segs == 'backward':
            implicit_outputs = {state_name, state_rate_name, initial_state_name}
        else:
            implicit_outputs = set()

        free_vars = free_vars - implicit_outputs

        if fix_initial:
            free_vars = free_vars - {initial_state_name}
        if fix_final:
            free_vars = free_vars - {final_state_name}

        if opt:
            # Add design variables for the remaining free variables
            if state_name in free_vars:
                self.add_design_var(name=state_name,
                                    lower=lower,
                                    upper=upper,
                                    scaler=scaler,
                                    adder=adder,
                                    ref0=ref0,
                                    ref=ref)

            if state_rate_name in free_vars:
                self.add_design_var(name=state_rate_name,
                                    scaler=scaler,
                                    adder=adder,
                                    ref0=ref0,
                                    ref=ref)

            if initial_state_name in free_vars:
                self.add_design_var(name=initial_state_name,
                                    lower=ib[0],
                                    upper=ib[1],
                                    scaler=scaler,
                                    adder=adder,
                                    ref0=ref0,
                                    ref=ref)

            if final_state_name in free_vars:
                self.add_design_var(name=final_state_name,
                                    lower=fb[0],
                                    upper=fb[1],
                                    scaler=scaler,
                                    adder=adder,
                                    ref0=ref0,
                                    ref=ref)

        return implicit_outputs

    def configure_io(self, phase):
        """
        This method is called during the owning phase's configure method since some aspects
        of the states, such as targets, are not known until phase configure.
        """
        collocation_comp = self._get_subsystem('collocation_comp')
        collocation_comp.configure_io(phase)

        gd = self.options['grid_data']
        nn = gd.subset_num_nodes['all']

        state_options = self.options['state_options']
        time_options = self.options['time_options']
        time_units = time_options['units']
        states_balance_comp = self._get_subsystem('states_balance_comp')

        for name, options in state_options.items():
            units = options['units']
            rate_source = options['rate_source']
            shape = options['shape']

            for tgt in options['targets']:
                self.promotes('ode_all', [(tgt, f'states:{name}')])
                self.set_input_defaults(f'states:{name}', val=1.0, units=units, src_shape=(nn,) + shape)

            implicit_outputs = self._configure_desvars(name, options)

            if f'states:{name}' in implicit_outputs:
                states_balance_comp.add_implicit_output(f'states:{name}', shape=(nn,) + shape, units=units,
                                                        resid_input=f'state_defects:{name}')

            if f'initial_states:{name}' in implicit_outputs:
                states_balance_comp.add_implicit_output(f'initial_states:{name}', shape=shape, units=units,
                                                        resid_input=f'final_state_defects:{name}')

            if f'final_states:{name}' in implicit_outputs:
                states_balance_comp.add_implicit_output(f'final_states:{name}', shape=shape, units=units,
                                                        resid_input=f'final_state_defects:{name}')

            if f'state_rates:{name}' in implicit_outputs:
                states_balance_comp.add_implicit_output(f'state_rates:{name}', shape=(nn,) + shape, units=units,
                                                        resid_input=f'state_rate_defects:{name}')

            try:
                rate_source_var = options['rate_source']
            except RuntimeError:
                raise ValueError(f"state '{state_name}' in phase '{phase.name}' was not given a "
                                 "rate_source")

            # Note the rate source must be shape-compatible with the state
            var_type = phase.classify_var(rate_source_var)

    def _get_rate_source_path(self, state_name, nodes, phase):
        """
        Return the rate source location and indices for a given state name.

        Parameters
        ----------
        state_name : str
            Name of the state.
        nodes : str
            One of ['col', 'all'].
        phase : dymos.Phase
            Phase object containing the rate source.

        Returns
        -------
        str
            Path to the rate source.
        ndarray
            Array of source indices.
        """
        gd = self.grid_data
        try:
            var = phase.state_options[state_name]['rate_source']
        except RuntimeError:
            raise ValueError(f"state '{state_name}' in phase '{phase.name}' was not given a "
                             "rate_source")

        # Note the rate source must be shape-compatible with the state
        var_type = phase.classify_var(var)

        # Determine the path to the variable
        if var_type == 't':
            rate_path = 't'
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 't_phase':
            rate_path = 't_phase'
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'state':
            rate_path = f'states:{var}'
            # Find the state_input indices which occur at segment endpoints, and repeat them twice
            state_input_idxs = gd.subset_node_indices['state_input']
            repeat_idxs = np.ones_like(state_input_idxs)
            if self.options['compressed']:
                segment_end_idxs = gd.subset_node_indices['segment_ends'][1:-1]
                # Repeat nodes that are on segment bounds (but not the first or last nodes in the phase)
                nodes_to_repeat = list(set(state_input_idxs).intersection(segment_end_idxs))
                # Now find these nodes in the state input indices
                idxs_of_ntr_in_state_inputs = np.where(np.in1d(state_input_idxs, nodes_to_repeat))[0]
                # All state input nodes are used once, but nodes_to_repeat are used twice
                repeat_idxs[idxs_of_ntr_in_state_inputs] = 2
            # Now we have a way of mapping the state input indices to all nodes
            map_input_node_idxs_to_all = np.repeat(np.arange(gd.subset_num_nodes['state_input'],
                                                             dtype=int), repeats=repeat_idxs)
            # Now select the subset of nodes we want to use.
            node_idxs = map_input_node_idxs_to_all[gd.subset_node_indices[nodes]]
        elif var_type == 'indep_control':
            rate_path = f'control_values:{var}'
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'input_control':
            rate_path = f'control_values:{var}'
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'control_rate':
            control_name = var[:-5]
            rate_path = f'control_rates:{control_name}_rate'
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'control_rate2':
            control_name = var[:-6]
            rate_path = f'control_rates:{control_name}_rate2'
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'indep_polynomial_control':
            rate_path = f'polynomial_control_values:{var}'
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'input_polynomial_control':
            rate_path = f'polynomial_control_values:{var}'
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'polynomial_control_rate':
            control_name = var[:-5]
            rate_path = f'polynomial_control_rates:{control_name}_rate'
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'polynomial_control_rate2':
            control_name = var[:-6]
            rate_path = f'polynomial_control_rates:{control_name}_rate2'
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'parameter':
            rate_path = f'parameter_vals:{var}'
            dynamic = not phase.parameter_options[var]['static_target']
            if dynamic:
                node_idxs = np.zeros(gd.subset_num_nodes[nodes], dtype=int)
            else:
                node_idxs = np.zeros(1, dtype=int)
        else:
            # Failed to find variable, assume it is in the ODE
            rate_path = f'ode_all.{var}'
            node_idxs = gd.subset_node_indices[nodes]

        src_idxs = om.slicer[node_idxs, ...]

        return rate_path, src_idxs