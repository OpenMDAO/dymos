__author__ = 'rfalck'

import os.path

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND


def setup_surrogates_all(model_name='CRM'):
    data_dir = os.path.split(os.path.realpath(__file__))[0]
    raw = np.loadtxt(os.path.join(data_dir, 'data', '{0}_aero_inputs.dat'.format(model_name)))
    [CL, CD, CM] = np.loadtxt(os.path.join(data_dir, 'data',
                                           '{0}_aero_outputs.dat'.format(model_name)))

    M_num, a_num, h_num, e_num = raw[:4].astype(int)
    M_surr = raw[4:4 + M_num]
    a_surr = raw[4 + M_num:4 + M_num + a_num]
    h_surr = raw[4 + M_num + a_num:4 + M_num + a_num + h_num]
    e_surr = raw[4 + M_num + a_num + h_num:]

    interp_CL = np.zeros((M_num, a_num, h_num, e_num))
    interp_CD = np.zeros((M_num, a_num, h_num, e_num))
    interp_CM = np.zeros((M_num, a_num, h_num, e_num))

    count = 0
    for i in range(M_num):
        for j in range(a_num):
            for k in range(h_num):
                for l in range(e_num):  # noqa: E741, allow ambiguous variable name 'l'
                    interp_CL[i][j][k][l] = CL[count]
                    interp_CD[i][j][k][l] = CD[count]
                    interp_CM[i][j][k][l] = CM[count]
                    count += 1

    interpND_CL = InterpND(method='lagrange3', points=(M_surr, a_surr, h_surr, e_surr),
                           values=interp_CL)
    interpND_CD = InterpND(method='lagrange3', points=(M_surr, a_surr, h_surr, e_surr),
                           values=interp_CD)
    interpND_CM = InterpND(method='lagrange3', points=(M_surr, a_surr, h_surr, e_surr),
                           values=interp_CM)

    nums = {
        'M': M_num,
        'a': a_num,
        'h': h_num,
        'e': e_num,
    }

    return [interpND_CL, interpND_CD, interpND_CM, nums]


class InterpNDAeroCoeffComp(om.ExplicitComponent):
    """ Compute the lift, drag, and moment coefficients of the aircraft """
    def initialize(self):
        self.options.declare('vec_size', types=int)
        self.options.declare('interpND_CL')
        self.options.declare('interpND_CD')
        self.options.declare('interpND_CM')
        self.options.declare('interp_num')

    def setup(self):
        nn = self.options['vec_size']

        # Inputs
        self.add_input(name='M', shape=(nn,), desc='Mach number', units=None)
        self.add_input(name='h', shape=(nn,), desc='Altitude', units='km')
        self.add_input(name='alpha', shape=(nn,), desc='Angle of attack', units='rad')
        self.add_input(name='eta', shape=(nn,), desc='tail rotation angle', units='rad')

        # Outputs
        self.add_output(name='CL', shape=(nn,), desc='Lift coefficient', units=None)
        self.add_output(name='CD', shape=(nn,), desc='Drag coefficient', units=None)
        self.add_output(name='CM', shape=(nn,), desc='Moment coefficient', units=None)

        # Initialization
        self.inputs = np.zeros((nn, 4))

        # interp_CL = self.options['interp_CL']
        interpND_CL = self.options['interpND_CL']
        interpND_CD = self.options['interpND_CD']
        interpND_CM = self.options['interpND_CM']

        self.interp_tup = ((0, 'CL', interpND_CL), (1, 'CD', interpND_CD), (2, 'CM', interpND_CM))

        ar = np.arange(nn)
        self.declare_partials('CL', 'M', rows=ar, cols=ar)
        self.declare_partials('CL', 'alpha', rows=ar, cols=ar)
        self.declare_partials('CL', 'h', rows=ar, cols=ar)
        self.declare_partials('CL', 'eta', rows=ar, cols=ar)

        self.declare_partials('CD', 'M', rows=ar, cols=ar)
        self.declare_partials('CD', 'alpha', rows=ar, cols=ar)
        self.declare_partials('CD', 'h', rows=ar, cols=ar)
        self.declare_partials('CD', 'eta', rows=ar, cols=ar)

        self.declare_partials('CM', 'M', rows=ar, cols=ar)
        self.declare_partials('CM', 'alpha', rows=ar, cols=ar)
        self.declare_partials('CM', 'h', rows=ar, cols=ar)
        self.declare_partials('CM', 'eta', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        self.inputs[:, 0] = inputs['M']
        self.inputs[:, 1] = np.degrees(inputs['alpha'])  # convert to deg
        self.inputs[:, 2] = inputs['h'] * 3.28e3   # convert km to ft
        self.inputs[:, 3] = np.degrees(inputs['eta'])  # convert to deg

        outputs['CL'][:] = self.options['interpND_CL'].interpolate(self.inputs)[:]
        outputs['CD'][:] = self.options['interpND_CD'].interpolate(self.inputs)[:] + 0.015
        outputs['CM'][:] = self.options['interpND_CM'].interpolate(self.inputs)[:]

    def compute_partials(self, inputs, partials):
        self.inputs[:, 0] = inputs['M']
        self.inputs[:, 1] = np.degrees(inputs['alpha'])  # convert to deg
        self.inputs[:, 2] = inputs['h'] * 3.28e3   # convert km to ft
        self.inputs[:, 3] = np.degrees(inputs['eta'])  # convert to deg

        for ind, name, interp in self.interp_tup:
            # compute_derivative
            values, derivs = interp.interpolate(self.inputs, compute_derivative=True)[:]
            partials[name, 'M'] = derivs[:, 0]
            partials[name, 'alpha'] = np.degrees(derivs[:, 1])
            partials[name, 'h'] = derivs[:, 2] * 3.28e3
            partials[name, 'eta'] = np.degrees(derivs[:, 3])
