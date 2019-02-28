"""Define the CollocationComp class."""

from __future__ import print_function, division, absolute_import

from numbers import Number
from six import string_types, iteritems

import numpy as np

from openmdao.api import ImplicitComponent, DirectSolver

from dymos.phases.grid_data import GridData
from dymos.utils.misc import get_rate_units
from dymos.utils.indexing import get_src_indices_by_row


class CollocationComp(ImplicitComponent):
    """
    A simple equation balance for solving implicit equations.

    Attributes
    ----------
    _state_vars : dict
        Cache the data provided during `add_balance`
        so everything can be saved until setup is called.
    """

    def initialize(self):

        self.options.declare(
            'grid_data', types=GridData,
            desc='Container object for grid info')

        self.options.declare(
            'state_options', types=dict,
            desc='Dictionary of state names/options for the phase')

        self.options.declare(
            'time_units', default=None, allow_none=True, types=string_types,
            desc='Units of time')

        # self.options.declare(
        #     'reverse_time', types=bool, default=False,
        #     desc='It True, the ODE integration happens backwards; from t_final to t_initial')

    def __init__(self, **kwargs):
        super(CollocationComp, self).__init__(**kwargs)

        self.state_idx_map = {}  # keyed by state_name, contains solver and optimizer index lists

        state_options = self.options['state_options']
        grid_data = self.options['grid_data']

        state_input = grid_data.subset_node_indices['state_input']

        # indecies into all_nodes that correspond to the solved and indep vars
        # (not accouting for fix_initial or fix_final)
        solver_solved = grid_data.subset_node_indices['solver_solved']
        solver_indep = grid_data.subset_node_indices['solver_indep']

        # numpy magic to find the locations in state_input that match the index-values
        # specified in solver_solved
        self.solver_node_idx = list(np.where(np.in1d(state_input, solver_solved))[0])
        self.indep_node_idx = list(np.where(np.in1d(state_input, solver_indep))[0])

        for state_name, options in iteritems(state_options):
            self.state_idx_map[state_name] = {'solver': None, 'indep': None}

            if options['solve_segments']:
                if options['fix_initial'] and options['fix_final']:
                    raise ValueError('Can not use solver based collocation defects '
                                     'with both "fix_initial" and "fix_final" turned on.')

                if (not options['fix_initial'] and not options['solve_continuity']) and \
                   (not options['fix_final']):
                    raise ValueError('Must have either fix_initial" and "fix_final" turned on '
                                     'with solver base collocation')

            if options['fix_initial'] or options['solve_continuity']:
                self.state_idx_map[state_name]['solver'] = self.solver_node_idx[1:]
                self.state_idx_map[state_name]['indep'] = \
                    [self.solver_node_idx[0]] + self.indep_node_idx

            elif options['fix_final']:
                self.state_idx_map[state_name]['solver'] = self.solver_node_idx[:-1]
                self.state_idx_map[state_name]['indep'] = \
                    self.indep_node_idx + [self.solver_node_idx[-1]]

        # NOTE: num_col_nodes MUST equal len(self.solver_node_idx) - 1 in order to ensure
        # you get a well defined problem; if that doesn't happen, something is wrong

    def setup(self):
        """
        Define the independent variables, output variables, and partials.
        """
        state_options = self.options['state_options']
        grid_data = self.options['grid_data']
        num_col_nodes = grid_data.subset_num_nodes['col']
        time_units = self.options['time_units']

        num_state_input_nodes = grid_data.subset_num_nodes['state_input']

        self.add_input('dt_dstau', units=time_units, shape=(num_col_nodes,))

        self.var_names = {}
        for state_name in state_options:
            self.var_names[state_name] = {
                'f_approx': 'f_approx:{0}'.format(state_name),
                'f_computed': 'f_computed:{0}'.format(state_name),
                'defect': 'defects:{0}'.format(state_name),
            }

        for state_name, options in iteritems(state_options):

            shape = options['shape']
            units = options['units']
            solved = options['solve_segments']
            var_names = self.var_names[state_name]

            rate_units = get_rate_units(units, time_units)

            # only need the implicit variable if this state is solved.
            # will get promoted to the same naming convention as the indepvar comp
            if solved:
                ref = 1.0 / options['defect_scaler']
                self.add_output(name='states:{0}'.format(state_name),
                                shape=(num_state_input_nodes, ) + shape,
                                units=units, ref=ref)

                # Input for continuity, which comes from an external balance when solved.
                if options['solve_continuity']:
                    self.add_input(name='initial_state_continuity:{0}'.format(state_name),
                                   shape=(1, ) + shape, units=units)

            self.add_input(
                name=var_names['f_approx'],
                shape=(num_col_nodes, ) + shape,
                desc='Estimated derivative of state {0} '
                     'at the collocation nodes'.format(state_name),
                units=rate_units)

            self.add_input(
                name=var_names['f_computed'],
                shape=(num_col_nodes, ) + shape,
                desc='Computed derivative of state {0} '
                     'at the collocation nodes'.format(state_name),
                units=rate_units)

            if not solved:
                # compute an output contraint value since the optimizer needs it
                self.add_output(
                    name=var_names['defect'],
                    shape=(num_col_nodes, ) + shape,
                    desc='Constraint value for interior defects of state {0}'.format(state_name),
                    units=units)

                if 'defect_ref' in options and options['defect_ref'] is not None:
                    def_scl = 1.0 / options['defect_ref']
                elif 'defect_scaler' in options:
                    def_scl = options['defect_scaler']
                else:
                    def_scl = 1.0

                self.add_constraint(name=var_names['defect'],
                                    equals=0.0,
                                    scaler=def_scl)

        # Setup partials
        for state_name, options in iteritems(state_options):
            shape = options['shape']
            size = np.prod(shape)
            solved = options['solve_segments']
            var_names = self.var_names[state_name]

            if solved:  # only need this deriv if its solved
                solve_idx = np.array(self.state_idx_map[state_name]['solver'])
                indep_idx = np.array(self.state_idx_map[state_name]['indep'])

                num_indep_nodes = indep_idx.shape[0]
                num_solve_nodes = solve_idx.shape[0]
                state_var_name = 'states:{0}'.format(state_name)

                base_idx = np.tile(np.arange(size), num_indep_nodes).reshape(num_indep_nodes, size)
                r = (indep_idx[:, np.newaxis]*size + base_idx).flatten()

                # anything that looks like an indep
                self.declare_partials(of=state_var_name, wrt=state_var_name,
                                      rows=r, cols=r, val=-1.0)

                if options['solve_continuity']:
                    initial_state_name = 'initial_state_continuity:{0}'.format(state_name)
                    self.declare_partials(of=state_var_name, wrt=initial_state_name,
                                          rows=r, cols=r, val=1.0)

                c = np.arange(num_solve_nodes * size)
                base_idx = np.tile(np.arange(size), num_solve_nodes).reshape(num_solve_nodes, size)
                r = (solve_idx[:, np.newaxis]*size + base_idx).flatten()

                self.declare_partials(of=state_var_name,
                                      wrt=var_names['f_approx'],
                                      rows=r, cols=c)

                self.declare_partials(of=state_var_name,
                                      wrt=var_names['f_computed'],
                                      rows=r, cols=c)

                c = np.repeat(np.arange(num_solve_nodes), size)
                self.declare_partials(of=state_var_name,
                                      wrt='dt_dstau',
                                      rows=r, cols=c)

            else:
                r = np.arange(num_col_nodes * size)
                defect_name = self.var_names[state_name]['defect']

                self.declare_partials(of=defect_name,
                                      wrt=defect_name,
                                      rows=r, cols=r, val=-1)

                self.declare_partials(of=defect_name,
                                      wrt=var_names['f_approx'],
                                      rows=r, cols=r)

                self.declare_partials(of=defect_name,
                                      wrt=var_names['f_computed'],
                                      rows=r, cols=r)

                c = np.repeat(np.arange(num_col_nodes), size)
                self.declare_partials(of=defect_name,
                                      wrt='dt_dstau',
                                      rows=r, cols=c)

    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        Calculate the residual for each balance.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        residuals : Vector
            unscaled, dimensional residuals written to via residuals[key]
        """
        state_options = self.options['state_options']
        dt_dstau = inputs['dt_dstau']

        for state_name, options in iteritems(state_options):
            var_names = self.var_names[state_name]

            f_approx = inputs[var_names['f_approx']]
            f_computed = inputs[var_names['f_computed']]

            if options['solve_segments']:
                solve_idx = self.state_idx_map[state_name]['solver']
                indep_idx = self.state_idx_map[state_name]['indep']
                state_var_name = 'states:{0}'.format(state_name)

                residuals[state_var_name][solve_idx, ...] = ((f_approx - f_computed).T * dt_dstau).T

                if options['solve_continuity']:
                    initial_state_name = 'initial_state_continuity:{0}'.format(state_name)
                    residuals[state_var_name][indep_idx, ...] = \
                        inputs[initial_state_name][indep_idx, ...] - \
                        outputs[state_var_name][indep_idx, ...]

                # really is: <idep_val> - \outputs[state_name][indep_idx] but OpenMDAO
                # implementation details mean we just set it to 0
                # but derivatives are still based on (<idep_val> - \outputs[state_name][indep_idx]),
                # so you get -1 wrt state var
                # NOTE: check_partials will report wrong derivs for the indep vars,
                #       but don't believe it!
                else:
                    residuals[state_var_name][indep_idx, ...] = 0

            else:
                residuals[var_names['defect']] = \
                    ((f_approx - f_computed).T * dt_dstau).T - outputs[var_names['defect']]

    def solve_nonlinear(self, inputs, outputs):
        state_options = self.options['state_options']
        dt_dstau = inputs['dt_dstau']

        for state_name, options in iteritems(state_options):
            if options['solve_segments']:
                if options['solve_continuity']:
                    output_name = 'states:{0}'.format(state_name)
                    input_name = 'initial_state_continuity:{0}'.format(state_name)
                    outputs[output_name][0, ...] = inputs[input_name]
            else:
                var_names = self.var_names[state_name]

                f_approx = inputs[var_names['f_approx']]
                f_computed = inputs[var_names['f_computed']]

                outputs[var_names['defect']] = ((f_approx - f_computed).T * dt_dstau).T

    def linearize(self, inputs, outputs, J):
        dt_dstau = inputs['dt_dstau']
        for state_name, options in iteritems(self.options['state_options']):
            size = np.prod(options['shape'])
            var_names = self.var_names[state_name]
            solved = options['solve_segments']
            f_approx = inputs[var_names['f_approx']]
            f_computed = inputs[var_names['f_computed']]

            k = np.repeat(dt_dstau, size)

            if solved:
                state_var_name = 'states:{0}'.format(state_name)
                J[state_var_name, var_names['f_approx']] = k
                J[state_var_name, var_names['f_computed']] = -k
                J[state_var_name, 'dt_dstau'] = (f_approx - f_computed).ravel()
            else:
                defect_name = self.var_names[state_name]['defect']
                J[defect_name, var_names['f_approx']] = k
                J[defect_name, var_names['f_computed']] = -k
                J[defect_name, 'dt_dstau'] = (f_approx - f_computed).ravel()

    # this mimics how the direct_solver works, for any dect outputs.
    # but I wonder if it might be faster to just use a direct solver
    # (or basically do a matrix equivalent operation)
    #     self.linear_solver = DirectSolver()
    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            for state_name, options in iteritems(self.options['state_options']):
                if not options['solve_segments']:
                    defect_name = self.var_names[state_name]['defect']
                    d_outputs[defect_name] = -d_residuals[defect_name]
        elif mode == 'rev':
            for state_name, options in iteritems(self.options['state_options']):
                if not options['solve_segments']:
                    defect_name = self.var_names[state_name]['defect']
                    d_residuals[defect_name] = -d_outputs[defect_name]
