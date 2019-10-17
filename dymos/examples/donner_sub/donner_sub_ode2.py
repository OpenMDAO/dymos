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
        self.add_input('time', shape=(nn,), units=None)
        self.add_output(f'ship_radius_{sub_index}', shape=(nn,), units=None)
        self.add_output(f'sub_{sub_index}_range', shape=(nn,), units=None)

        ar = np.arange(nn, dtype=int)
        self.declare_partials(of=f'ship_radius_{sub_index}', wrt='lat', rows=ar, cols=ar)
        self.declare_partials(of=f'ship_radius_{sub_index}', wrt='lon', rows=ar, cols=ar)
        self.declare_partials(of=f'sub_{sub_index}_range', wrt='lat', rows=ar, cols=ar)
        self.declare_partials(of=f'sub_{sub_index}_range', wrt='lon', rows=ar, cols=ar)
        self.declare_partials(of=f'sub_{sub_index}_range', wrt='time', rows=ar, cols=ar)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        lat = inputs['lat']
        lon = inputs['lon']
        time = inputs['time']

        sub_index = self.options['sub_index']
        sub_origin = self.options['sub_origin']

        outputs[f'ship_radius_{sub_index}'] = (lat - sub_origin[1])**2 + (lon - sub_origin[0])**2
        outputs[f'sub_{sub_index}_range'] = outputs[f'ship_radius_{sub_index}'] - inputs['time'] ** 2

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        sub_index = self.options['sub_index']

        lat = inputs['lat']
        lon = inputs['lon']
        time = inputs['time']

        partials[f'ship_radius_{sub_index}', 'lat'] = 2 * lat
        partials[f'ship_radius_{sub_index}', 'lon'] = 2 * lon
        partials[f'sub_{sub_index}_range', 'lat'] = 2 * lat
        partials[f'sub_{sub_index}_range', 'lon'] = 2 * lon
        partials[f'sub_{sub_index}_range', 'time'] = -2 * time


class DonnerSubODE(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_subs', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        nsub = self.options['num_subs']

        self.add_subsystem('ship_eom', ShipEOMComp(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])

        for i in range(nsub):
            sub_origin_lat = -1 + (i + 1) * 2 / (nsub + 1)
            self.add_subsystem(f'ship_rad_{i}',
                               ShipRadiusComp(num_nodes=nn, sub_index=i,
                                              sub_origin=(sub_origin_lat, 0)),
                               promotes_inputs=['*'], promotes_outputs=['*'])
