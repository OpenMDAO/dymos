import numpy as np
import openmdao.api as om

from .birkhoff_collocation_comp import BirkhoffCollocationComp
from .birkhoff_state_resid_comp import BirkhoffStateResidComp

from ...grid_data import GridData
from ....utils.misc import get_rate_units, CoerceDesvar, reshape_val
from ....utils.lagrange import lagrange_matrices
from ....utils.indexing import get_desvar_indices
from ....utils.constants import INF_BOUND
from ...._options import options as dymos_options
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
        self.options.declare('grid_data', types=GridData, desc='Container object for grid info for the control inputs.')
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

        self.add_subsystem('ode', subsys=ode_class(num_nodes=nn, **ode_init_kwargs))

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

        ib = options['initial_bounds']
        fb = options['final_bounds']
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

    def configure_io(self):
        """
        This method is called during the owning phase's configure method since some aspects
        of the states, such as targets, are not known until phase configure.
        """
        collocation_comp = self._get_subsystem('collocation_comp')
        collocation_comp.configure_io()

        gd = self.options['grid_data']
        nn = gd.subset_num_nodes['all']

        state_options = self.options['state_options']
        time_options = self.options['time_options']
        time_units = time_options['units']
        states_balance_comp = self._get_subsystem('states_balance_comp')

        for name, options in state_options.items():
            units = options['units']
            rate_source = options['rate_source']
            rate_units = get_rate_units(units, time_units)
            shape = options['shape']

            for tgt in options['targets']:
                self.promotes('ode', [(tgt, f'states:{name}')])
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
                states_balance_comp.add_implicit_output(f'state_rates:{name}', shape=(nn,) + shape, units=rate_units,
                                                        resid_input=f'state_rate_defects:{name}')

            self.connect(f'ode.{rate_source}', f'f_computed:{name}')
