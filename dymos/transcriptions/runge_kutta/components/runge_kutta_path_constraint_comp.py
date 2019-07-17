from __future__ import print_function, division, absolute_import

import numpy as np

from ...common.path_constraint_comp import PathConstraintCompBase


class RungeKuttaPathConstraintComp(PathConstraintCompBase):

    def setup(self):
        """
        Define the independent variables as output variables.
        """
        num_nodes = self.options['num_nodes']

        for (name, kwargs) in self._path_constraints:

            input_kwargs = {k: kwargs[k] for k in ('units', 'desc')}
            input_name = 'all_values:{0}'.format(name)
            self.add_input(input_name,
                           shape=(num_nodes,) + kwargs['shape'],
                           **input_kwargs)

            output_name = 'path:{0}'.format(name)
            output_kwargs = {k: kwargs[k] for k in ('units', 'desc')}
            output_kwargs['shape'] = (num_nodes,) + kwargs['shape']
            self.add_output(output_name, **output_kwargs)

            constraint_kwargs = {k: kwargs.get(k, None)
                                 for k in ('lower', 'upper', 'equals', 'ref', 'ref0', 'adder',
                                           'scaler', 'indices', 'linear')}

            # Convert indices from those in one time instance to those in all time instances
            template = np.zeros(np.prod(kwargs['shape']), dtype=int)
            template[kwargs['indices']] = 1
            template = np.tile(template, num_nodes)
            constraint_kwargs['indices'] = np.nonzero(template)[0]

            self.add_constraint(output_name, **constraint_kwargs)

            self._vars.append((input_name, output_name, kwargs['shape']))

            # Setup partials

            all_shape = (num_nodes,) + kwargs['shape']
            var_size = np.prod(kwargs['shape'])
            all_size = np.prod(all_shape)

            all_row_starts = np.arange(num_nodes, dtype=int) * var_size
            all_rows = []
            for i in all_row_starts:
                all_rows.extend(range(i, i + var_size))
            all_rows = np.asarray(all_rows, dtype=int)

            self.declare_partials(
                of=output_name,
                wrt=input_name,
                dependent=True,
                rows=all_rows,
                cols=np.arange(all_size),
                val=1.0)

    def compute(self, inputs, outputs):
        for (input_name, output_name, _) in self._vars:
            outputs[output_name] = inputs[input_name]
