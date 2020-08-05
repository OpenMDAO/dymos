import numpy as np

import openmdao.api as om


class WaterEngine(om.Group):
    """
    Computes thrust and water flow for a water.

    Simplifications:
     - the pressure due to the water column in the non inertial frame (i.e.
       under a+g acceleration) is insignificant compared to the air pressure
     - the water does not have appreciable speed inside the bottle
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='water_exhaust_speed',
                           subsys=_WaterExhaustSpeed(num_nodes=nn),
                           promotes=['p', 'p_a', 'rho_w'])

        self.add_subsystem(name='water_flow_rate',
                           subsys=_WaterFlowRate(num_nodes=nn),
                           promotes=['A_out', 'Vdot'])

        self.add_subsystem(name='pressure_rate',
                           subsys=_PressureRate(num_nodes=nn),
                           promotes=['p', 'k', 'V_b', 'V_w', 'Vdot', 'pdot'])

        self.add_subsystem(name='water_thrust',
                           subsys=_WaterThrust(num_nodes=nn),
                           promotes=['rho_w', 'A_out', 'F'])

        self.connect('water_exhaust_speed.v_out', 'water_flow_rate.v_out')
        self.connect('water_exhaust_speed.v_out', 'water_thrust.v_out')


class _WaterExhaustSpeed(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='rho_w', val=1e3*np.ones(nn), desc='water density', units='kg/m**3')
        self.add_input(name='p', val=6.5e5*np.ones(nn), desc='air pressure', units='N/m**2')  # 5.5bar = 80 psi
        self.add_input(name='p_a', val=1.01e5*np.ones(nn), desc='air pressure', units='N/m**2')

        self.add_output(name='v_out', shape=(nn,), desc='water exhaust speed', units='m/s')

        ar = np.arange(nn)

        self.declare_partials(of='*', wrt='*', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        p = inputs['p']
        p_a = inputs['p_a']
        rho_w = inputs['rho_w']

        outputs['v_out'] = np.sqrt(2*(p-p_a)/rho_w)

    def compute_partials(self, inputs, partials):
        p = inputs['p']
        p_a = inputs['p_a']
        rho_w = inputs['rho_w']

        v_out = np.sqrt(2*(p-p_a)/rho_w)

        partials['v_out', 'p'] = 1/v_out/rho_w
        partials['v_out', 'p_a'] = -1/v_out/rho_w
        partials['v_out', 'rho_w'] = dv_outdrho_w = 1/v_out*(-(p-p_a)/rho_w**2)


class _WaterFlowRate(om.ExplicitComponent):
    """ Computer water flow rate"""
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='A_out', val=np.ones(nn), desc='nozzle outlet area', units='m**2')
        self.add_input(name='v_out', val=np.zeros(nn), desc='water exhaust speed', units='m/s')

        self.add_output(name='Vdot', shape=(nn,), desc='water flow', units='m**3/s')

        ar = np.arange(nn)

        self.declare_partials(of='*', wrt='*', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        A_out = inputs['A_out']
        v_out = inputs['v_out']

        outputs['Vdot'] = -v_out*A_out

    def compute_partials(self, inputs, partials):
        A_out = inputs['A_out']
        v_out = inputs['v_out']

        partials['Vdot', 'A_out'] = -v_out
        partials['Vdot', 'v_out'] = -A_out


class _PressureRate(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='p', val=np.ones(nn), desc='air pressure', units='N/m**2')
        self.add_input(name='k', val=1.4*np.ones(nn), desc='polytropic coefficient for expansion', units=None)
        self.add_input(name='V_b', val=2e-3*np.ones(nn), desc='bottle volume', units='m**3')
        self.add_input(name='V_w', val=1e-3*np.ones(nn), desc='water volume', units='m**3')
        self.add_input(name='Vdot', shape=(nn,), desc='water flow', units='m**3/s')

        self.add_output(name='pdot', shape=(nn,), desc='pressure derivative', units='N/m**2/s')

        ar = np.arange(nn)

        self.declare_partials(of='*', wrt='*', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        p = inputs['p']
        k = inputs['k']
        V_b = inputs['V_b']
        V_w = inputs['V_w']
        Vdot = inputs['Vdot']

        pdot = p*k*Vdot/(V_b-V_w)

        outputs['pdot'] = pdot

    def compute_partials(self, inputs, partials):
        p = inputs['p']
        k = inputs['k']
        V_b = inputs['V_b']
        V_w = inputs['V_w']
        Vdot = inputs['Vdot']

        partials['pdot', 'p'] = k*Vdot/(V_b-V_w)
        partials['pdot', 'k'] = p*Vdot/(V_b-V_w)
        partials['pdot', 'V_b'] = -p*Vdot/(V_b-V_w)**2
        partials['pdot', 'V_w'] = p*Vdot/(V_b-V_w)**2
        partials['pdot', 'Vdot'] = p*k/(V_b-V_w)


class _WaterThrust(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='rho_w', val=1e3*np.ones(nn), desc='water density', units='kg/m**3')
        self.add_input(name='A_out', val=np.pi*13e-3**2/4*np.ones(nn), desc='nozzle outlet area', units='m**2')
        self.add_input(name='v_out', val=np.zeros(nn), desc='water exhaust speed', units='m/s')

        self.add_output(name='F', shape=(nn,), desc='thrust', units='N')

        ar = np.arange(nn)

        self.declare_partials(of='*', wrt='*', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        rho_w = inputs['rho_w']
        A_out = inputs['A_out']
        v_out = inputs['v_out']

        outputs['F'] = rho_w*v_out**2*A_out

    def compute_partials(self, inputs, partials):
        rho_w = inputs['rho_w']
        A_out = inputs['A_out']
        v_out = inputs['v_out']

        partials['F', 'A_out'] = rho_w*v_out**2
        partials['F', 'rho_w'] = v_out**2*A_out
        partials['F', 'v_out'] = 2*rho_w*v_out*A_out


if __name__ == '__main__':
    p = om.Problem()
    p.model = WaterEngine(num_nodes=1)
    p.setup()
    p.check_config(checks=['unconnected_inputs'], out_file=None)
    p.final_setup()
