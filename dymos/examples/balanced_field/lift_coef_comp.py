import numpy as np
import openmdao.api as om


class LiftCoefComp(om.ExplicitComponent):
    """
    Compute an aircraft lift-coefficient given four values:
    - current angle of attack (alpha)
    - angle of attack at maximum CL (alpha_max)
    - zero-alpha angle of attack (CL0)
    - maximum lift coefficient (CL_max)
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('alpha', val=np.ones(nn), desc='angle of attack', units='deg')
        self.add_input('CL0', val=0.5, desc='zero-alpha lift coefficient', units=None)
        self.add_input('CL_max', val=2.0, desc='maximum lift coefficient', units=None)
        self.add_input('alpha_max', val=10, desc='angle of attack at CL_max', units='deg')

        self.add_output('CL', val=np.ones(nn), desc='lift coefficient', units=None)

        self.declare_coloring(wrt='*', method='cs', tol=1.0E-12, show_sparsity=False, show_summary=False)
        self.declare_partials('CL', '*', method='cs')

    def compute(self, inputs, outputs):
        CL0 = inputs['CL0']
        alpha = inputs['alpha']
        alpha_max = inputs['alpha_max']
        CL_max = inputs['CL_max']

        outputs['CL'] = CL0 + (alpha / alpha_max) * (CL_max - CL0)


if __name__ == '__main__':  # pragma: no cover
    import matplotlib.pyplot as plt

    nn = 20
    p = om.Problem()
    p.model.add_subsystem('cl_comp', LiftCoefComp(num_nodes=nn))

    p.setup()

    p.set_val('cl_comp.alpha', np.linspace(0, 14, nn))

    p.run_model()

    p.check_partials(compact_print=True)

    import matplotlib.pyplot as plt
    plt.plot(p.get_val('cl_comp.alpha'), p.get_val('cl_comp.CL'))
    plt.show()
