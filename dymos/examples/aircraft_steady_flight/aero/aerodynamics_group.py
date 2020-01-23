import openmdao.api as om
from .aero_coef_comp import AeroCoefComp
from .aero_forces_comp import AeroForcesComp
from .mbi_aero_coef_comp import MBIAeroCoeffComp, setup_surrogates_all

try:
    import MBI
except ImportError:
    MBI = None


class AerodynamicsGroup(om.Group):
    """
    The purpose of the Aerodynamics is to compute the lift and
    drag forces on the aircraft.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        n = self.options['num_nodes']

        if MBI is None:
            self.add_subsystem(name='aero_coef_comp',
                               subsys=AeroCoefComp(vec_size=n, extrapolate=True,
                                                   method='lagrange3'),
                               promotes_inputs=['mach', 'alpha', 'alt', 'eta'],
                               promotes_outputs=['CL', 'CD', 'CM'])

        else:
            mbi_CL, mbi_CD, mbi_CM, mbi_num = setup_surrogates_all()
            mbi_CL.seterr(bounds='warn')
            mbi_CD.seterr(bounds='warn')
            mbi_CM.seterr(bounds='warn')

            self.add_subsystem(name='aero_coef_comp',
                               subsys=MBIAeroCoeffComp(vec_size=n, mbi_CL=mbi_CL, mbi_CD=mbi_CD,
                                                       mbi_CM=mbi_CM, mbi_num=mbi_num),
                               promotes_inputs=[('M', 'mach'), 'alpha', ('h', 'alt'), 'eta'],
                               promotes_outputs=['CL', 'CD', 'CM'])

        self.add_subsystem(name='aero_forces_comp',
                           subsys=AeroForcesComp(num_nodes=n),
                           promotes_inputs=['q', 'S', 'CL', 'CD'],
                           promotes_outputs=['L', 'D'])
