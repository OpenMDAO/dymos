import openmdao.api as om

from openmdao.components.input_resids_comp import InputResidsComp

from .radau_defect_comp import RadauDefectComp

from dymos.transcriptions.grid_data import GridData
from dymos.phase.options import TimeOptionsDictionary
from dymos.utils.ode_utils import _make_ode_system
from dymos.utils.misc import broadcast_to_nodes, determine_ref_ref0


class RadauIterGroup(om.Group):
    """
    Class definition for the RadauIterGroup.

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
        """Declare group options."""
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
        self.options.declare('calc_exprs', types=dict, default={}, recordable=False,
                             desc='Calculation expressions of the parent phase.')
        self.options.declare('parameter_options', types=dict, default={},
                             desc='Parameter options for the phase.')

    def setup(self):
        """
        Define the structure of the RadauIterGroup.
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

        self.add_subsystem('defects',
                           subsys=RadauDefectComp(grid_data=gd,
                                                  state_options=state_options,
                                                  time_units=time_options['units']),
                           promotes_inputs=['*'], promotes_outputs=['*'])

        if any([opts['solve_segments'] in ('forward', 'backward') for opts in state_options.values()]):
            self.add_subsystem('states_resids_comp', subsys=InputResidsComp(),
                               promotes_inputs=['*'], promotes_outputs=['*'])

    def _configure_desvars(self, name, options):
        state_name = f'states:{name}'
        initial_state_name = f'initial_states:{name}'
        final_state_name = f'final_states:{name}'
        state_rate_name = f'state_rates:{name}'

        solve_segs = options['solve_segments']
        opt = options['opt']

        num_input_nodes = self.options['grid_data'].subset_num_nodes['state_input']

        ib = (None, None) if options['initial_bounds'] is None else options['initial_bounds']
        fb = (None, None) if options['final_bounds'] is None else options['final_bounds']
        lower = options['lower']
        upper = options['upper']

        scaler = options['scaler']
        adder = options['adder']
        ref0 = options['ref0']
        ref = options['ref']

        ref0, ref = determine_ref_ref0(ref0, ref, adder, scaler)
        scaler = None
        adder = None

        fix_initial = options['fix_initial']
        fix_final = options['fix_final']
        input_initial = options['input_initial']
        input_final = options['input_final']
        shape = options['shape']

        if not isinstance(fix_initial, bool):
            raise ValueError(f'Option fix_initial for state {name} must be True or False. '
                             f'If fixing some indices of a non-scalar state, use initial '
                             f'boundary constraints.')
        if not isinstance(fix_final, bool):
            raise ValueError(f'Option fix_final for state {name} must be True or False. '
                             f'If fixing some indices of a non-scalar state, use '
                             f'final boundary constraints.')

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

        ref0_at_input_nodes = broadcast_to_nodes(ref0, shape, num_input_nodes)
        ref_at_input_nodes = broadcast_to_nodes(ref, shape, num_input_nodes)

        free_vars = {state_name, initial_state_name, final_state_name}

        if solve_segs == 'forward':
            implicit_outputs = {state_name, final_state_name}
        elif solve_segs == 'backward':
            implicit_outputs = {state_name, initial_state_name}
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
                                    ref0=ref0_at_input_nodes,
                                    ref=ref_at_input_nodes)

            if state_rate_name in free_vars:
                self.add_design_var(name=state_rate_name,
                                    ref0=ref0_at_input_nodes,
                                    ref=ref_at_input_nodes)

            if initial_state_name in free_vars:
                self.add_design_var(name=initial_state_name,
                                    lower=ib[0],
                                    upper=ib[1],
                                    ref0=ref0,
                                    ref=ref)

            if final_state_name in free_vars:
                self.add_design_var(name=final_state_name,
                                    lower=fb[0],
                                    upper=fb[1],
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
        defect_comp = self._get_subsystem('defects')
        defect_comp.configure_io(phase)

        gd = self.options['grid_data']
        nn = gd.subset_num_nodes['all']
        nin = gd.subset_num_nodes['state_input']
        ncn = gd.subset_num_nodes['col']
        ns = gd.num_segments
        state_src_idxs_input_to_disc = gd.input_maps['state_input_to_disc']
        state_src_idxs_input_to_all = state_src_idxs_input_to_disc

        col_idxs = gd.subset_node_indices['col']

        state_options = self.options['state_options']
        states_resids_comp = self._get_subsystem('states_resids_comp')

        self.promotes('defects', inputs=('dt_dstau',),
                      src_indices=om.slicer[col_idxs, ...],
                      src_shape=(nn,))

        for name, options in state_options.items():
            units = options['units']
            rate_source = options['rate_source']
            shape = options['shape']

            for tgt in options['targets']:
                self.promotes('ode_all', [(tgt, f'states:{name}')],
                              src_indices=om.slicer[state_src_idxs_input_to_all, ...])

            self.set_input_defaults(f'states:{name}', val=1.0, units=units, src_shape=(nin,) + shape)

            self.promotes('defects', inputs=(f'states:{name}',),
                          src_indices=om.slicer[state_src_idxs_input_to_all, ...],)

            self._implicit_outputs = self._configure_desvars(name, options)

            if f'states:{name}' in self._implicit_outputs:
                # ref0 = options['ref0'] if options['ref0'] is not None else 0.0
                # ref = options['ref'] if options['ref'] is not None else 1.0

                states_resids_comp.add_input(f'initial_state_defects:{name}', shape=(1,) + shape, units=units)
                states_resids_comp.add_input(f'final_state_defects:{name}', shape=(1,) + shape, units=units)
                states_resids_comp.add_input(f'state_rate_defects:{name}', shape=(ncn,) + shape, units=units)

                if ns > 1 and not gd.compressed:
                    states_resids_comp.add_input(f'state_cnty_defects:{name}',
                                                 shape=(ns - 1,) + shape,
                                                 units=units)
                    # For noncompressed transcription, resids provides values at overlapping
                    # segment boundaries.
                    # TODO: Something is wrong with providing ref at all the nodes for this output in OpenMDAO.
                    # ref0_at_nodes = broadcast_to_nodes(ref0, shape, nn)
                    # ref_at_nodes = broadcast_to_nodes(ref, shape, nn)
                    # defect_ref_at_nodes = ref_at_nodes

                    states_resids_comp.add_output(f'states:{name}',
                                                  shape=(nn,) + shape,
                                                  lower=options['lower'],
                                                  upper=options['upper'],
                                                  #   ref0=ref0_at_nodes,
                                                  #   ref=ref_at_nodes,
                                                  #   res_ref=defect_ref_at_nodes,
                                                  units=units)
                else:
                    # For compressed transcription, resids comp provides values at input nodes.
                    nin = gd.subset_num_nodes['state_input']

                    # TODO: Something is wrong with providing ref at all the nodes for this output in OpenMDAO.
                    # ref0_at_input_nodes = broadcast_to_nodes(ref0, shape, nin)
                    # ref_at_input_nodes = broadcast_to_nodes(ref, shape, nin)
                    # defect_ref_at_input_nodes = ref_at_input_nodes

                    states_resids_comp.add_output(f'states:{name}',
                                                  shape=(nin,) + shape,
                                                  lower=options['lower'],
                                                  upper=options['upper'],
                                                  #   ref0=ref0_at_input_nodes,
                                                  #   ref=ref_at_input_nodes,
                                                  #   res_ref=defect_ref_at_input_nodes,
                                                  units=units)

            if options['initial_bounds'] is None:
                initial_lb = options['lower']
                initial_ub = options['upper']
            else:
                initial_lb, initial_ub = options['initial_bounds']

            if options['final_bounds'] is None:
                final_lb = options['lower']
                final_ub = options['upper']
            else:
                final_lb, final_ub = options['final_bounds']

            # TODO: add scaling
            if f'initial_states:{name}' in self._implicit_outputs:
                states_resids_comp.add_output(f'initial_states:{name}', shape=(1,) + shape, units=units,
                                              lower=initial_lb, upper=initial_ub)

            if f'final_states:{name}' in self._implicit_outputs:
                states_resids_comp.add_output(f'final_states:{name}', shape=(1,) + shape, units=units,
                                              lower=final_lb, upper=final_ub)

            try:
                rate_source_var = options['rate_source']
            except RuntimeError:
                raise ValueError(f"state '{name}' in phase '{phase.name}' was not given a "
                                 "rate_source")

            # Note the rate source must be shape-compatible with the state
            var_type = phase.classify_var(rate_source_var)

            if var_type == 'ode':
                self.connect(f'ode_all.{rate_source}', f'f_ode:{name}',
                             src_indices=om.slicer[gd.subset_node_indices['col'], ...])
