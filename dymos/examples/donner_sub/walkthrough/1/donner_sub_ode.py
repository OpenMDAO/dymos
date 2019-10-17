import numpy as np
import openmdao.api as om


class ShipEOMComp(om.ExplicitComponent):

    def initialize(self):

        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('speed', shape=(nn,), units=None)
        self.add_input('heading', shape=(nn,), units='rad')
        self.add_output('dlon_dt', shape=(nn,), units=None)
        self.add_output('dlat_dt', shape=(nn,), units=None)

        ar = np.arange(nn, dtype=int)
        self.declare_partials(of='dlon_dt', wrt='speed', rows=ar, cols=ar)
        self.declare_partials(of='dlon_dt', wrt='heading', rows=ar, cols=ar)
        self.declare_partials(of='dlat_dt', wrt='speed', rows=ar, cols=ar)
        self.declare_partials(of='dlat_dt', wrt='heading', rows=ar, cols=ar)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        v = inputs['speed']
        hdg = inputs['heading']

        outputs['dlon_dt'] = v * np.sin(hdg)
        outputs['dlat_dt'] = v * np.cos(hdg)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        v = inputs['speed']
        hdg = inputs['heading']
        partials['dlon_dt', 'speed'] = np.sin(hdg)
        partials['dlon_dt', 'heading'] = v * np.cos(hdg)
        partials['dlat_dt', 'speed'] = np.cos(hdg)
        partials['dlat_dt', 'heading'] = -v * np.sin(hdg)


class SubRadiusComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('time', shape=(nn,), units=None)
        self.add_output('sub_radius', shape=(nn,), units=None)

        ar = np.arange(nn, dtype=int)
        self.declare_partials(of='sub_radius', wrt='time', rows=ar, cols=ar)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        t = inputs['time']
        outputs['sub_radius'] = t**2

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        t = inputs['time']
        partials['sub_radius', 'time'] = 2 * t


class ShipRadiusComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('sub_index', types=int, default=0)
        self.options.declare('sub_origin', types=tuple, default=(0, 0))

    def setup(self):
        nn = self.options['num_nodes']
        sub_index = self.options['sub_index']

        self.add_input('lat', shape=(nn,), units=None)
        self.add_input('lon', shape=(nn,), units=None)
        self.add_output('r_ship', shape=(nn,), units=None)
        self.add_output(f'sub_{sub_index}_range', shape=(nn,), units=None)

        ar = np.arange(nn, dtype=int)
        self.declare_partials(of='r_ship', wrt='lat', rows=ar, cols=ar)
        self.declare_partials(of='r_ship', wrt='lon', rows=ar, cols=ar)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        lat = inputs['lat']
        lon = inputs['lon']

        sub_origin = self.options['sub_origin']

        outputs['r_ship'] = np.sqrt((lat - sub_origin[1])**2 + (lon - sub_origin[0])**2)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        sub_origin = self.options['sub_origin']

        lat = inputs['lat']
        lon = inputs['lon']

        r = np.sqrt((lat - sub_origin[1])**2 + (lon - sub_origin[0])**2)

        partials['r_ship', 'lat'] = lat / r
        partials['r_ship', 'lon'] = lon / r


class DonnerSubODE(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('ship_eom', ShipEOMComp(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('ship_rad', ShipRadiusComp(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('exec_comp', om.ExecComp('sub_range = r_ship - time',
                                                    sub_range={'shape': (nn,)},
                                                    r_ship={'shape': (nn,)},
                                                    time={'shape': (nn,)},
                                                    has_diag_partials=True),
                           promotes_inputs=['*'], promotes_outputs=['*'])
