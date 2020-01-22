import numpy as np
import openmdao.api as om

from .crm_data import h_bp, alpha_bp, mach_bp, eta_bp, CL_data, CD_data, CM_data


class AeroCoefComp(om.MetaModelStructuredComp):
    """ Interpolates aerodynamic coefficients for the NASA Common Research Model. """

    def setup(self):
        nn = self.options['vec_size']
        self.add_input(name='mach', val=0.2 * np.ones(nn), units=None, training_data=mach_bp)
        self.add_input(name='alpha', val=0.0 * np.ones(nn), units='deg', training_data=alpha_bp)
        self.add_input(name='alt', val=0.0 * np.ones(nn), units='ft', training_data=h_bp)
        self.add_input(name='eta', val=0.0 * np.ones(nn), units='deg', training_data=eta_bp)

        self.add_output(name='CL', val=np.zeros(nn), units=None, training_data=CL_data)
        self.add_output(name='CD', val=np.zeros(nn), units=None, training_data=CD_data + 0.015)
        self.add_output(name='CM', val=np.zeros(nn), units=None, training_data=CM_data)
