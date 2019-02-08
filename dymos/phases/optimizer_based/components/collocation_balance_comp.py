"""Define the BalanceComp class."""

from __future__ import print_function, division, absolute_import

from numbers import Number
from six import string_types, iteritems

import numpy as np

from openmdao.core.implicitcomponent import ImplicitComponent

from dymos.phases.grid_data import GridData
from dymos.utils.misc import get_rate_units
from dymos.utils.indexing import get_src_indices_by_row

class CollocationBalanceComp(ImplicitComponent):
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


    def setup(self):
        """
        Define the independent variables, output variables, and partials.
        """
        state_options = self.options['state_options']
        grid_data = self.options['grid_data']
        num_col_nodes = grid_data.subset_num_nodes['col']
        time_units = self.options['time_units']

        num_state_input_nodes = grid_data.subset_num_nodes['state_input']

        seg_ends = grid_data.subset_node_indices['segment_ends']
        state_input = grid_data.subset_node_indices['state_input']

        # indecies into all_nodes that correspond to the solved and indep vars (not accouting for fix_initial or fix_final)
        solver_solved = grid_data.subset_node_indices['solver_solved']
        solver_indep = grid_data.subset_node_indices['solver_indep']

        # numpy magic to find the locations in state_input that match the index-values specified in solver_solved
        self.solver_node_idx = list(np.where(np.in1d(state_input, solver_solved))[0])
        self.indep_node_idx = list(np.where(np.in1d(state_input, solver_indep))[0])

        # NOTE: num_col_nodes MUST equal len(self.solver_node_idx) - 1 in order to ensure you get a well defined problem
        #       if that doesn't happen, something is wrong

        self.state_idx_map = {} # keyed by state_name, contains solver and optimizer index lists        
   
        for state_name, options in iteritems(state_options):
            self.state_idx_map[state_name] = {'solver':None, 'indep':None}
            if options['fix_initial'] and options['fix_final']: 
                raise ValueError('Can not use solver based collocation defects with both "fix_initial" and "fix_final" turned on.')

            if (not options['fix_initial']) and (not options['fix_final']): 
                raise ValueError('Must have either fix_initial" and "fix_final" turned on with solver base collocation')

            elif options['fix_initial']: 
                self.state_idx_map[state_name]['solver'] = self.solver_node_idx[1:]
                self.state_idx_map[state_name]['indep'] = [self.solver_node_idx[0]] + self.indep_node_idx

            elif options['fix_final']: 
                self.state_idx_map[state_name]['solver'] = self.solver_node_idx[:-1]
                self.state_idx_map[state_name]['indep'] = self.indep_node_idx + [self.solver_node_idx[-1]]


        self.add_input('dt_dstau', units=time_units, 
                       shape=(num_col_nodes,))

        self.var_names = var_names = {}
        for state_name in state_options:
            var_names[state_name] = {
                'f_approx': 'f_approx:{0}'.format(state_name),
                'f_computed': 'f_computed:{0}'.format(state_name),
            }

        for state_name, options in iteritems(state_options):

            shape = options['shape']
            units = options['units']

            rate_units = get_rate_units(units, time_units)

            self.add_output(name=state_name,
                            shape=(num_state_input_nodes,) + shape,
                            units=units)

            self.add_input(
                name=var_names[state_name]['f_approx'],
                shape=(num_col_nodes,) + shape,
                desc='Estimated derivative of state {0} '
                     'at the collocation nodes'.format(state_name),
                units=rate_units)

            self.add_input(
                name=var_names[state_name]['f_computed'],
                shape=(num_col_nodes,) + shape,
                desc='Computed derivative of state {0} '
                     'at the collocation nodes'.format(state_name),
                units=rate_units)


        # Setup partials

        for state_name, options in iteritems(state_options):
            shape = options['shape']
            size = np.prod(shape)

            solve_idx = np.array(self.state_idx_map[state_name]['solver'])
            indep_idx = np.array(self.state_idx_map[state_name]['indep'])
            num_indep_nodes = indep_idx.shape[0]
            num_solve_nodes = solve_idx.shape[0]
           
            base_idx = np.tile(np.arange(size),num_indep_nodes).reshape(num_indep_nodes,size) 
            r = (indep_idx[:,np.newaxis]*size + base_idx).flatten()
            self.declare_partials(of=state_name, wrt=state_name, 
                                 rows=r, cols=r, val=-1)
 
            c = np.arange(num_solve_nodes * size)
            base_idx = np.tile(np.arange(size),num_solve_nodes).reshape(num_solve_nodes,size) 
            r = (solve_idx[:,np.newaxis]*size + base_idx).flatten()

            var_names = self.var_names[state_name]
            self.declare_partials(of=state_name,
                                  wrt=var_names['f_approx'],
                                  rows=r, cols=c)

            self.declare_partials(of=state_name,
                                  wrt=var_names['f_computed'],
                                  rows=r, cols=c)

            c = np.repeat(np.arange(num_solve_nodes), size)
            self.declare_partials(of=state_name,
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

        for state_name in state_options:
            # print(state_name)
            
            var_names = self.var_names[state_name]

            f_approx = inputs[var_names['f_approx']]
            f_computed = inputs[var_names['f_computed']]

            # IndepVarComp residuals are always 0 

            solve_idx = self.state_idx_map[state_name]['solver']
            indep_idx = self.state_idx_map[state_name]['indep']

            residuals[state_name][solve_idx,...] = ((f_approx - f_computed).T * dt_dstau).T
            
            # really is: <idep_val> - \outputs[state_name][indep_idx] but OpenMDAO implementation details mean we just set it to 0
            # but derivatives are based on <idep_val> - \outputs[state_name][indep_idx], so you get -1 wrt state var
            # NOTE: Because of this weirdness check_partials will report wrong derivs for the indep vars, but don't believe it!
            residuals[state_name][indep_idx,...] = 0 

    def linearize(self, inputs, outputs, J):
        dt_dstau = inputs['dt_dstau']
        for state_name, options in iteritems(self.options['state_options']):
            size = np.prod(options['shape'])
            var_names = self.var_names[state_name]
            f_approx = inputs[var_names['f_approx']]
            f_computed = inputs[var_names['f_computed']]

            k = np.repeat(dt_dstau, size)

            J[state_name, var_names['f_approx']] = k
            J[state_name, var_names['f_computed']] = -k
            J[state_name, 'dt_dstau'] = (f_approx - f_computed).ravel()

