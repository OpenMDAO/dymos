"""Define the StateIndependents class."""

import numpy as np

import openmdao.api as om

from ....transcriptions.grid_data import GridData
from ....options import options as dymos_options


class StateIndependentsComp(om.ImplicitComponent):
    """
    A simple component that replaces the state indepvarcomps whenver the solver needs to solve for
    the state or whenever the initial state is connected to an external source.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_check_partials = not dymos_options['include_check_partials']

    def initialize(self):

        self.options.declare(
            'grid_data', types=GridData,
            desc='Container object for grid info')

        self.options.declare(
            'state_options', types=dict,
            desc='Dictionary of state names/options for the phase')

    def configure_io(self, state_idx_map):
        """
        I/O creation is delayed until configure so that we can determine the shape and units for
        the states.
        """
        self.state_idx_map = state_idx_map
        state_options = self.options['state_options']
        grid_data = self.options['grid_data']
        num_col_nodes = grid_data.subset_num_nodes['col']

        num_state_input_nodes = grid_data.subset_num_nodes['state_input']

        self.var_names = {}
        for state_name in state_options:
            self.var_names[state_name] = {
                'defect': f'defects:{state_name}',
            }

        for state_name, options in state_options.items():

            shape = options['shape']
            units = options['units']
            solved = options['solve_segments']
            var_names = self.var_names[state_name]

            # only need the implicit variable if this state is solved.
            # Note: we don't add scaling and bounds here. This may be revisited.
            self.add_output(name=f'states:{state_name}',
                            shape=(num_state_input_nodes, ) + shape,
                            units=units)

            # Input for continuity, which can come from an external source.
            if options['connected_initial']:
                input_name = f'initial_states:{state_name}',
                self.add_input(name=input_name, shape=(1, ) + shape, units=units)

            # compute an output contraint value since the optimizer needs it
            if solved:
                self.add_input(
                    name=var_names['defect'],
                    shape=(num_col_nodes, ) + shape,
                    desc=f'Constraint value for interior defects of state {state_name}',
                    units=units)

        # Setup partials
        for state_name, options in state_options.items():
            shape = options['shape']
            size = np.prod(shape)
            solved = options['solve_segments']
            state_var_name = f'states:{state_name}'

            if solved:  # only need this deriv if its solved
                solve_idx = np.array(state_idx_map[state_name]['solver'])
                indep_idx = np.array(state_idx_map[state_name]['indep'])

                num_indep_nodes = indep_idx.shape[0]
                num_solve_nodes = solve_idx.shape[0]

                base_idx = np.tile(np.arange(size), num_indep_nodes).reshape(num_indep_nodes, size)
                row = (indep_idx[:, np.newaxis]*size + base_idx).flatten()

                # anything that looks like an indep
                self.declare_partials(of=state_var_name, wrt=state_var_name,
                                      rows=row, cols=row, val=-1.0)

                if options['connected_initial']:
                    wrt = f'initial_states:{state_name}'
                    row_col = np.arange(np.prod(shape))
                    self.declare_partials(of=state_var_name, wrt=wrt, rows=row_col, cols=row_col,
                                          val=1.0)

                col = np.arange(num_solve_nodes * size)
                base_idx = np.tile(np.arange(size), num_solve_nodes).reshape(num_solve_nodes, size)
                row = (solve_idx[:, np.newaxis]*size + base_idx).flatten()

                var_names = self.var_names[state_name]
                self.declare_partials(of=state_var_name,
                                      wrt=var_names['defect'],
                                      rows=row, cols=col, val=1.0)

            else:
                row_col = np.arange(num_state_input_nodes*np.prod(shape))
                self.declare_partials(of=state_var_name, wrt=state_var_name,
                                      rows=row_col, cols=row_col, val=-1.0)

                if options['connected_initial']:
                    wrt = f'initial_states:{state_name}'
                    row_col = np.arange(np.prod(shape))
                    self.declare_partials(of=state_var_name, wrt=wrt, rows=row_col, cols=row_col,
                                          val=1.0)

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

        for state_name, options in state_options.items():

            state_var_name = f'states:{state_name}'

            solve_idx = self.state_idx_map[state_name]['solver']
            indep_idx = self.state_idx_map[state_name]['indep']

            var_names = self.var_names[state_name]
            defect = inputs[var_names['defect']]

            residuals[state_var_name][solve_idx, ...] = defect

            # really is: <idep_val> - \outputs[state_name][indep_idx] but OpenMDAO
            # implementation details mean we just set it to 0
            # but derivatives are still based on (<idep_val> - \outputs[state_name][indep_idx]),
            # so you get -1 wrt state var
            # NOTE: check_partials will report wrong derivs for the indep vars,
            #       but don't believe it!
            residuals[state_var_name][indep_idx, ...] = 0.0

            if options['connected_initial']:
                ic_state_name = f'initial_states:{state_name}'

                residuals[state_var_name][0, ...] = \
                    inputs[ic_state_name][0, ...] - \
                    outputs[state_var_name][0, ...]

    def solve_nonlinear(self, inputs, outputs):
        state_options = self.options['state_options']

        for state_name, options in state_options.items():
            if options['connected_initial']:
                output_name = f'states:{state_name}'
                input_name = f'initial_states:{state_name}'
                outputs[output_name][0, ...] = inputs[input_name]
