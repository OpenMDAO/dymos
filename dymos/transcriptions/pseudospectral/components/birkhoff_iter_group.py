import numpy as np
import openmdao.api as om

from .birkhoff_defect_comp import BirkhoffDefectComp

from ...grid_data import GridData
from ....phase.options import TimeOptionsDictionary
from ....utils.ode_utils import _make_ode_system


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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._implicit_outputs = set()

    def initialize(self):
        """
        Declare group options.
        """
        self.options.declare('state_options', types=dict,
                             desc='Dictionary of options for the states.')
        self.options.declare('time_options', types=TimeOptionsDictionary,
                             desc='Options for time in the phase.')
        self.options.declare('parameter_options', types=dict, default={},
                             desc='parameter options inherited from the phase')
        self.options.declare('grid_data', types=GridData, desc='Container object for grid info.')
        self.options.declare('ode_class', default=None,
                             desc='Callable that instantiates the ODE system.',
                             recordable=False)
        self.options.declare('ode_init_kwargs', types=dict, default={},
                             desc='Keyword arguments provided when initializing the ODE System')
        self.options.declare('calc_exprs', types=dict, default={},
                             desc='ODE Expresions from the Phase')

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

        ode = _make_ode_system(ode_class=ode_class,
                               num_nodes=nn,
                               ode_init_kwargs=ode_init_kwargs,
                               calc_exprs=self.options['calc_exprs'],
                               parameter_options=self.options['parameter_options'])

        self.add_subsystem('ode_all', subsys=ode)

        self.add_subsystem('collocation_comp',
                           subsys=BirkhoffDefectComp(grid_data=gd,
                                                     state_options=state_options,
                                                     time_units=time_options['units']),
                           promotes_inputs=['*'], promotes_outputs=['*'])

        if any([opts['solve_segments'] in ('forward', 'backward') for opts in state_options.values()]):
            self.add_subsystem('states_balance_comp', subsys=om.InputResidsComp(),
                               promotes_inputs=['*'], promotes_outputs=['*'])

    def _configure_desvars(self, name, options):
        state_name = f'states:{name}'
        initial_state_name = f'initial_states:{name}'
        final_state_name = f'final_states:{name}'
        state_rate_name = f'state_rates:{name}'

        solve_segs = options['solve_segments']
        opt = options['opt']

        num_nodes = self.options['grid_data'].subset_num_nodes['col']

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
        input_initial = options['input_initial']
        input_final = options['input_final']
        shape = options['shape']

        if solve_segs == 'forward' and fix_final:
            raise ValueError(f"Option fix_final on state {name} may not "
                             f"be used with `solve_segments='forward'`.\n Use "
                             f"a boundary constraint to constrain the final "
                             f"state value instead.")
        elif solve_segs == 'backward' and fix_initial:
            raise ValueError(f"Option fix_final on state {name} may not "
                             f"be used with `solve_segments='forward'`.\n Use "
                             f"a boundary constraint to constrain the initial "
                             f"state value instead.")

        if not np.isscalar(ref0) and ref0 is not None:
            ref0 = np.asarray(ref0)
            if ref0.shape == shape:
                ref0_state = np.tile(ref0.flatten(), num_nodes)
                # ref0_seg_ends = np.tile(ref0.flatten(), 2)
            else:
                raise ValueError('array-valued scaler/ref must length equal to state-size')
        else:
            ref0_state = ref0
            # ref0_seg_ends = ref0
        if not np.isscalar(ref) and ref is not None:
            ref = np.asarray(ref)
            if ref.shape == shape:
                ref_state = np.tile(ref.flatten(), num_nodes)
                # ref_seg_ends = np.tile(ref.flatten(), 2)
            else:
                raise ValueError('array-valued scaler/ref must length equal to state-size')
        else:
            ref_state = ref
            # ref_seg_ends = ref

        free_vars = {state_name, state_rate_name, initial_state_name, final_state_name}

        if solve_segs == 'forward':
            implicit_outputs = {state_name, state_rate_name, final_state_name}
        elif solve_segs == 'backward':
            implicit_outputs = {state_name, state_rate_name, initial_state_name}
        else:
            implicit_outputs = set()

        free_vars = free_vars - implicit_outputs

        if fix_initial or input_initial:
            free_vars = free_vars - {initial_state_name}
        if fix_final or input_final:
            free_vars = free_vars - {final_state_name}

        if opt:
            # Add design variables for the remaining free variables
            if state_name in free_vars:
                self.add_design_var(name=state_name,
                                    lower=lower,
                                    upper=upper,
                                    scaler=scaler,
                                    adder=adder,
                                    ref0=ref0_state,
                                    ref=ref_state)

            if state_rate_name in free_vars:
                self.add_design_var(name=state_rate_name,
                                    scaler=scaler,
                                    adder=adder,
                                    ref0=ref0_state,
                                    ref=ref_state)

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
        I/O creation is delayed until configure so that we can determine shape and units for the states.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        collocation_comp = self._get_subsystem('collocation_comp')
        collocation_comp.configure_io(phase)

        gd = self.options['grid_data']
        nn = gd.subset_num_nodes['all']
        ns = gd.num_segments

        state_options = self.options['state_options']
        states_balance_comp = self._get_subsystem('states_balance_comp')

        for name, options in state_options.items():
            units = options['units']
            rate_source = options['rate_source']
            shape = options['shape']

            for tgt in options['targets']:
                self.promotes('ode_all', [(tgt, f'states:{name}')])
                self.set_input_defaults(f'states:{name}', val=1.0, units=units, src_shape=(nn,) + shape)

            self._implicit_outputs = self._configure_desvars(name, options)

            if f'states:{name}' in self._implicit_outputs:
                states_balance_comp.add_output(f'states:{name}',
                                               shape=(nn,) + shape,
                                               units=units)

                states_balance_comp.add_input(f'state_defects:{name}',
                                              shape=(nn+ns,) + shape,
                                              units=units)

                if ns > 1:
                    states_balance_comp.add_input(f'state_cnty_defects:{name}',
                                                  shape=(ns - 1,) + shape,
                                                  units=units)

            if f'state_rates:{name}' in self._implicit_outputs:
                states_balance_comp.add_output(f'state_rates:{name}', shape=(nn,) + shape, units=units)
                states_balance_comp.add_input(f'state_rate_defects:{name}',
                                              shape=(nn,) + shape,
                                              units=units)

            if f'initial_states:{name}' in self._implicit_outputs:
                states_balance_comp.add_output(f'initial_states:{name}', shape=(1,) + shape, units=units)

            if f'final_states:{name}' in self._implicit_outputs:
                states_balance_comp.add_output(f'final_states:{name}', shape=(1,) + shape, units=units)

            try:
                rate_source_var = options['rate_source']
            except RuntimeError:
                raise ValueError(f"state '{name}' in phase '{phase.name}' was not given a "
                                 "rate_source")

            # Note the rate source must be shape-compatible with the state
            var_type = phase.classify_var(rate_source_var)

            if var_type == 'ode':
                self.connect(f'ode_all.{rate_source}', f'f_computed:{name}', src_indices=om.slicer[...])
