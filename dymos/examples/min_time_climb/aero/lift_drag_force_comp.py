import numpy as np

import openmdao.api as om


class LiftDragForceComp(om.ExplicitComponent):
    """
    Compute the aerodynamic forces on the vehicle in the wind axis frame
    (lift, drag, cross) force.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='CL', val=np.zeros(nn,), desc='lift coefficient', units=None)
        self.add_input(name='CD', val=np.zeros(nn,), desc='drag coefficient', units=None)
        self.add_input(name='q', val=np.zeros(nn,), desc='dynamic pressure', units='N/m**2')
        self.add_input(name='S', val=np.zeros(nn,), desc='aerodynamic reference area', units='m**2')

        self.add_output(name='f_lift', shape=(nn,), desc='aerodynamic lift force', units='N')
        self.add_output(name='f_drag', shape=(nn,), desc='aerodynamic drag force', units='N')

        ar = np.arange(nn)

        self.declare_partials(of='f_lift', wrt='q', dependent=True, rows=ar, cols=ar)
        self.declare_partials(of='f_lift', wrt='S', dependent=True, rows=ar, cols=ar)
        self.declare_partials(of='f_lift', wrt='CL', dependent=True, rows=ar, cols=ar)

        self.declare_partials(of='f_drag', wrt='q', dependent=True, rows=ar, cols=ar)
        self.declare_partials(of='f_drag', wrt='S', dependent=True, rows=ar, cols=ar)
        self.declare_partials(of='f_drag', wrt='CD', dependent=True, rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        q = inputs['q']
        S = inputs['S']
        CL = inputs['CL']
        CD = inputs['CD']

        qS = q * S

        outputs['f_lift'] = qS * CL
        outputs['f_drag'] = qS * CD

    def compute_partials(self, inputs, partials):
        q = inputs['q']
        S = inputs['S']
        CL = inputs['CL']
        CD = inputs['CD']

        qS = q * S

        partials['f_lift', 'q'] = S * CL
        partials['f_lift', 'S'] = q * CL
        partials['f_lift', 'CL'] = qS

        partials['f_drag', 'q'] = S * CD
        partials['f_drag', 'S'] = q * CD
        partials['f_drag', 'CD'] = qS
