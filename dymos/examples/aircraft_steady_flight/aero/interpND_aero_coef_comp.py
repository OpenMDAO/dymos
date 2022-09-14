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

    mbi_CL = np.zeros((M_num, a_num, h_num, e_num))
    mbi_CD = np.zeros((M_num, a_num, h_num, e_num))
    mbi_CM = np.zeros((M_num, a_num, h_num, e_num))

    count = 0
    for i in range(M_num):
        for j in range(a_num):
            for k in range(h_num):
                for l in range(e_num):
                    mbi_CL[i][j][k][l] = CL[count]
                    mbi_CD[i][j][k][l] = CD[count]
                    mbi_CM[i][j][k][l] = CM[count]
                    count += 1

    interpND_CL = InterpND(method='lagrange3', points=(M_surr, a_surr, h_surr, e_surr), values=mbi_CL)
    interpND_CD = InterpND(method='lagrange3', points=(M_surr, a_surr, h_surr, e_surr), values=mbi_CD)
    interpND_CM = InterpND(method='lagrange3', points=(M_surr, a_surr, h_surr, e_surr), values=mbi_CM)

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
        self.options.declare('mbi_num')

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

        # mbi_CL = self.options['mbi_CL']
        interpND_CL = self.options['interpND_CL']
        interpND_CD = self.options['interpND_CD']
        interpND_CM = self.options['interpND_CM']

        self.mbi_tup = ((0, 'CL', interpND_CL), (1, 'CD', interpND_CD), (2, 'CM', interpND_CM))

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

        for ind, name, mbi in self.mbi_tup:
            # compute_derivative

            if name != 'CqqqqqM':
            # if name != 'CM':

                # data = mbi.evaluate(self.inputs, 1, 0)[:, 0]
                values, derivs = mbi.interpolate(self.inputs, compute_derivative=True)[:]
                # partials[name, 'M'] = data
                partials[name, 'M'] = derivs[:,0]

                # data = mbi.interpolate(self.inputs, compute_derivative=True)[:]
                data = derivs[:,1]
                data = np.degrees(data)
                partials[name, 'alpha'] = data

                # data = np.degrees(mbi.evaluate(self.inputs, 2, 0)[:, 0])
                # partials[name, 'alpha'] = data

                partials[name, 'h'] = derivs[:,2] * 3.28e3

                # data = mbi.evaluate(self.inputs, 3, 0)[:, 0] * 3.28e3
                # partials[name, 'h'] = data

                # data = np.degrees(mbi.evaluate(self.inputs, 4, 0)[:, 0])
                # partials[name, 'eta'] = data

                data = derivs[:,3]
                data = np.degrees(data)
                partials[name, 'eta'] = data
            else:
                data = mbi.evaluate(self.inputs, 1, 0)[:, 0]
                partials[name, 'M'] = data

                data = np.degrees(mbi.evaluate(self.inputs, 2, 0)[:, 0])
                partials[name, 'alpha'] = data

                data = mbi.evaluate(self.inputs, 3, 0)[:, 0] * 3.28e3
                partials[name, 'h'] = data

                data = np.degrees(mbi.evaluate(self.inputs, 4, 0)[:, 0])
                partials[name, 'eta'] = data

