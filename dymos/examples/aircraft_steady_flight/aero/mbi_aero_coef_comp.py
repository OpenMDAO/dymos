from __future__ import print_function, division

__author__ = 'rfalck'

import os.path

import numpy as np
from openmdao.api import ExplicitComponent

try:
    import MBI
except ImportError:
    print("MBI is not available")
    MBI = None

def setup_surrogates_all(model_name):
    data_dir = os.path.split(os.path.realpath(__file__))[0]
    raw = np.loadtxt(os.path.join(data_dir,'data','{0}_aero_inputs.dat'.format(model_name)))
    [CL, CD, CM] = np.loadtxt(os.path.join(data_dir,'data','{0}_aero_outputs.dat'.format(model_name)))

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

    CL_arr = MBI.MBI(mbi_CL, [M_surr, a_surr, h_surr, e_surr],
                             [M_num,  a_num,  h_num,  e_num],
                             [4, 4, 4, 4])
    CD_arr = MBI.MBI(mbi_CD, [M_surr, a_surr, h_surr, e_surr],
                             [M_num,  a_num,  h_num,  e_num],
                             [4, 4, 4, 4])
    CM_arr = MBI.MBI(mbi_CM, [M_surr, a_surr, h_surr, e_surr],
                             [M_num,  a_num,  h_num,  e_num],
                             [4, 4, 4, 4])

    nums = {
        'M': M_num,
        'a': a_num,
        'h': h_num,
        'e': e_num,
    }

    return [CL_arr, CD_arr, CM_arr, nums]


class MBIAeroCoeffComp(ExplicitComponent):
    """ Compute the lift, drag, and moment coefficients of the aircraft """
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('mbi_CL')
        self.options.declare('mbi_CD')
        self.options.declare('mbi_CM')
        self.options.declare('mbi_num')

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input(name='M',shape=(nn,), desc='Mach number', units=None)
        self.add_input(name='h',shape=(nn,), desc='Altitude', units='km')
        self.add_input(name='alpha',shape=(nn,), desc='Angle of attack', units='rad')
        self.add_input(name='eta',shape=(nn,), desc='tail rotation angle', units='rad')

        # Outputs
        self.add_output(name='C_L',shape=(nn,), desc='Lift coefficient', units=None)
        self.add_output(name='C_D',shape=(nn,), desc='Drag coefficient', units=None)
        self.add_output(name='C_M',shape=(nn,), desc='Moment coefficient', units=None)

        # Jacobian
        #self.J = {}

        # Initialization
        self.inputs = np.zeros((nn,4))
        #self.mbi_CL,self.mbi_CD,self.mbi_CM,self.mbi_num = setup_surrogates_all(aircraft_model)
        # self.mbi_CL,self.mbi_CD,self.mbi_CM,self.mbi_num = mbi_CL,mbi_CD,mbi_CM,mbi_num

        mbi_CL = self.options['mbi_CL']
        mbi_CD = self.options['mbi_CD']
        mbi_CM = self.options['mbi_CM']
        mbi_num = self.options['mbi_num']

        self.mbi_tup = ((0,'C_L',mbi_CL),(1,'C_D',mbi_CD),(2,'C_M',mbi_CM))

        ar = np.arange(nn)
        self.declare_partials('C_L', 'M', rows=ar, cols=ar)
        self.declare_partials('C_L', 'alpha', rows=ar, cols=ar)
        self.declare_partials('C_L', 'h', rows=ar, cols=ar)
        self.declare_partials('C_L', 'eta', rows=ar, cols=ar)

        self.declare_partials('C_D', 'M', rows=ar, cols=ar)
        self.declare_partials('C_D', 'alpha', rows=ar, cols=ar)
        self.declare_partials('C_D', 'h', rows=ar, cols=ar)
        self.declare_partials('C_D', 'eta', rows=ar, cols=ar)

        self.declare_partials('C_M', 'M', rows=ar, cols=ar)
        self.declare_partials('C_M', 'alpha', rows=ar, cols=ar)
        self.declare_partials('C_M', 'h', rows=ar, cols=ar)
        self.declare_partials('C_M', 'eta', rows=ar, cols=ar)


    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']

        self.inputs[:, 0] = inputs['M']
        self.inputs[:, 1] = np.degrees(inputs['alpha']) # convert to deg
        self.inputs[:, 2] = inputs['h'] * 3.28e3   # convert km to ft
        self.inputs[:, 3] = np.degrees(inputs['eta']) # convert to deg

        outputs['C_L'][:] = self.options['mbi_CL'].evaluate(self.inputs)[:, 0]
        outputs['C_D'][:] = self.options['mbi_CD'].evaluate(self.inputs)[:, 0] + 0.015
        outputs['C_M'][:] = self.options['mbi_CM'].evaluate(self.inputs)[:, 0]

    def compute_partials(self, inputs, partials):

        self.inputs[:, 0] = inputs['M']
        self.inputs[:, 1] = np.degrees(inputs['alpha']) # convert to deg
        self.inputs[:, 2] = inputs['h'] * 3.28e3   # convert km to ft
        self.inputs[:, 3] = np.degrees(inputs['eta']) # convert to deg

        for ind,name,mbi in self.mbi_tup:

            data = mbi.evaluate(self.inputs, 1, 0)[:,0]
            partials[name, 'M'] = data

            data = np.degrees(mbi.evaluate(self.inputs, 2, 0)[:,0])
            partials[name, 'alpha'] = data

            data = mbi.evaluate(self.inputs, 3, 0)[:,0] * 3.28e3
            partials[name, 'h'] = data

            data = np.degrees(mbi.evaluate(self.inputs, 4, 0)[:,0])
            partials[name, 'eta'] = data
