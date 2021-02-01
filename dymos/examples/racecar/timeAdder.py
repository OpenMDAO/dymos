import openmdao.api as om
import numpy as np


class TimeAdder(om.ExplicitComponent):
    # This serves to keep track of the elapsed time. Since we performed a change of variables we
    # need to integrate time by adding dt/ds

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # states
        self.add_input('sdot', val=np.zeros(nn), desc='distance along track', units='m/s')

        # outputs
        self.add_output('dt_ds', val=np.zeros(nn), desc='distance perpendicular to centerline',
                        units='s/m')

        # Setup partials
        arange = np.arange(self.options['num_nodes'], dtype=int)

        # partials
        self.declare_partials(of='dt_ds', wrt='sdot', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        sdot = inputs['sdot']

        outputs['dt_ds'] = 1/sdot

    def compute_partials(self, inputs, jacobian):
        sdot = inputs['sdot']

        jacobian['dt_ds', 'sdot'] = -1/sdot**2
