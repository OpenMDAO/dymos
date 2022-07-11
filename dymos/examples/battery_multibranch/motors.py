"""
Simple model for a set of motors in parallel where efficiency is a function of current.
"""
import numpy as np

import openmdao.api as om


class Motors(om.ExplicitComponent):
    """
    Model for motors in parallel.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1)
        self.options.declare('n_parallel', default=3, desc='number of motors in parallel')

    def setup(self):
        num_nodes = self.options['num_nodes']

        # Inputs
        self.add_input('power_out_gearbox', val=3.6*np.ones(num_nodes), units='W',
                       desc='Power at gearbox output')
        self.add_input('current_in_motor', val=np.ones(num_nodes), units='A',
                       desc='Total current demanded')

        # Outputs
        self.add_output('power_in_motor', val=np.ones(num_nodes), units='W',
                        desc='Power required at motor input')

        # Derivatives
        row_col = np.arange(num_nodes)

        self.declare_partials(of='power_in_motor', wrt=['*'], rows=row_col, cols=row_col)

    def compute(self, inputs, outputs):
        current = inputs['current_in_motor']
        power_out = inputs['power_out_gearbox']
        n_parallel = self.options['n_parallel']

        # Simple linear curve fit for efficiency.
        eff = 0.9 - 0.3 * current / n_parallel

        outputs['power_in_motor'] = power_out / eff

    def compute_partials(self, inputs, partials):
        current = inputs['current_in_motor']
        power_out = inputs['power_out_gearbox']
        n_parallel = self.options['n_parallel']

        eff = 0.9 - 0.3 * current / n_parallel

        partials['power_in_motor', 'power_out_gearbox'] = 1.0 / eff
        partials['power_in_motor', 'current_in_motor'] = 0.3 * power_out / (n_parallel * eff**2)


if __name__ == '__main__':

    import openmdao.api as om
    num_nodes = 1

    prob = om.Problem(model=Motors(num_nodes=num_nodes))
    model = prob.model

    prob.setup()

    prob.run_model()

    derivs = prob.check_partials(compact_print=True)

    print('done')
