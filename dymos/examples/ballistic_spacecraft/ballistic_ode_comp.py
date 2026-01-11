import jax.numpy as jnp
import openmdao.api as om


class BallisticODEComp(om.JaxExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('mu', types=float)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('r', shape=(nn, 3), units='km')
        self.add_input('v', shape=(nn, 3), units='km/s')
        self.add_output('r_dot', shape=(nn, 3), units='km/s')
        self.add_output('v_dot', shape=(nn, 3), units='km/s**2')

    def get_self_statics(self):
        return (self.options['mu'],)

    def compute_primal(self, r, v):
        r_mag = jnp.linalg.norm(r, axis=-1, keepdims=True)
        r_dot = v
        v_dot = -self.options['mu'] * r / r_mag ** 3
        return r_dot, v_dot
