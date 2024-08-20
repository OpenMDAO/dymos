import openmdao.api as om
from .aero_forces_comp import AeroForcesComp
from .interpND_aero_coef_comp import InterpNDAeroCoeffComp, setup_surrogates_all


class AerodynamicsGroup(om.Group):
    """
    The purpose of the Aerodynamics is to compute the lift and
    drag forces on the aircraft.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        n = self.options['num_nodes']

        interpND_CL, interpND_CD, interpND_CM, interp_num = setup_surrogates_all()

        self.add_subsystem(name='aero_coef_comp',
                           subsys=InterpNDAeroCoeffComp(vec_size=n, interpND_CL=interpND_CL, interpND_CD=interpND_CD,
                                                        interpND_CM=interpND_CM,
                                                        interp_num=interp_num),
                           promotes_inputs=[('M', 'mach'), 'alpha', ('h', 'alt'), 'eta'],
                           promotes_outputs=['CL', 'CD', 'CM'])

        self.add_subsystem(name='aero_forces_comp',
                           subsys=AeroForcesComp(num_nodes=n),
                           promotes_inputs=['q', 'S', 'CL', 'CD'],
                           promotes_outputs=['L', 'D'])
