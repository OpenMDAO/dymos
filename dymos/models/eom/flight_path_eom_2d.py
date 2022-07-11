import numpy as np

import openmdao.api as om


class FlightPathEOM2D(om.ExplicitComponent):
    """
    Component containing the ODE for 2D flight.

    Computes the position and velocity equations of motion using a 2D flight path
    parameterization of states per equations 4.42 - 4.46 of _[1].

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.

    References
    ----------
    .. [1] Bryson, Arthur Earl. Dynamic optimization. Vol. 1. Prentice Hall, p.172, 1999.
    """
    def initialize(self):
        """
        Declare component options.
        """
        self.options.declare('num_nodes', types=int)

    def setup(self):
        """
        Add inputs and outputs to this component.
        """
        nn = self.options['num_nodes']

        self.add_input(name='m',
                       val=np.ones(nn),
                       desc='aircraft mass',
                       units='kg')

        self.add_input(name='v',
                       val=np.ones(nn),
                       desc='aircraft velocity magnitude',
                       units='m/s')

        self.add_input(name='T',
                       val=np.zeros(nn),
                       desc='thrust',
                       units='N')

        self.add_input(name='alpha',
                       val=np.zeros(nn),
                       desc='angle of attack',
                       units='rad')

        self.add_input(name='L',
                       val=np.zeros(nn),
                       desc='lift force',
                       units='N')

        self.add_input(name='D',
                       val=np.zeros(nn),
                       desc='drag force',
                       units='N')

        self.add_input(name='gam',
                       val=np.zeros(nn),
                       desc='flight path angle',
                       units='rad')

        self.add_output(name='v_dot',
                        val=np.zeros(nn),
                        desc='rate of change of velocity magnitude',
                        units='m/s**2')

        self.add_output(name='gam_dot',
                        val=np.zeros(nn),
                        desc='rate of change of flight path angle',
                        units='rad/s')

        self.add_output(name='h_dot',
                        val=np.zeros(nn),
                        desc='rate of change of altitude',
                        units='m/s')

        self.add_output(name='r_dot',
                        val=np.zeros(nn),
                        desc='rate of change of range',
                        units='m/s')

        ar = np.arange(nn)

        self.declare_partials('v_dot', 'T', rows=ar, cols=ar)
        self.declare_partials('v_dot', 'D', rows=ar, cols=ar)
        self.declare_partials('v_dot', 'm', rows=ar, cols=ar)
        self.declare_partials('v_dot', 'gam', rows=ar, cols=ar)
        self.declare_partials('v_dot', 'alpha', rows=ar, cols=ar)

        self.declare_partials('gam_dot', 'T', rows=ar, cols=ar)
        self.declare_partials('gam_dot', 'L', rows=ar, cols=ar)
        self.declare_partials('gam_dot', 'm', rows=ar, cols=ar)
        self.declare_partials('gam_dot', 'gam', rows=ar, cols=ar)
        self.declare_partials('gam_dot', 'alpha', rows=ar, cols=ar)
        self.declare_partials('gam_dot', 'v', rows=ar, cols=ar)

        self.declare_partials('h_dot', 'gam', rows=ar, cols=ar)
        self.declare_partials('h_dot', 'v', rows=ar, cols=ar)

        self.declare_partials('r_dot', 'gam', rows=ar, cols=ar)
        self.declare_partials('r_dot', 'v', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        """
        Compute ODE outputs.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        """
        g = 9.80665
        m = inputs['m']
        v = inputs['v']
        T = inputs['T']
        L = inputs['L']
        D = inputs['D']
        gam = inputs['gam']
        alpha = inputs['alpha']

        calpha = np.cos(alpha)
        salpha = np.sin(alpha)

        cgam = np.cos(gam)
        sgam = np.sin(gam)

        mv = m * v

        outputs['v_dot'] = (T * calpha - D) / m - g * sgam

        outputs['gam_dot'] = (T * salpha + L) / mv - (g / v) * cgam

        outputs['h_dot'] = v * sgam

        outputs['r_dot'] = v * cgam

    def compute_partials(self, inputs, partials):
        """
        Compute sub-jacobian parts. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        partials : Jacobian
            Subjac components written to partials[output_name, input_name].
        """
        g = 9.80665
        m = inputs['m']
        v = inputs['v']
        T = inputs['T']
        L = inputs['L']
        D = inputs['D']
        gam = inputs['gam']
        alpha = inputs['alpha']

        calpha = np.cos(alpha)
        salpha = np.sin(alpha)

        cgam = np.cos(gam)
        sgam = np.sin(gam)

        mv = m * v

        partials['v_dot', 'T'] = calpha / m
        partials['v_dot', 'D'] = -1.0 / m
        partials['v_dot', 'm'] = (D - T * calpha) / (m**2)
        partials['v_dot', 'gam'] = -g * cgam
        partials['v_dot', 'alpha'] = -T * salpha / m

        partials['gam_dot', 'T'] = salpha / mv
        partials['gam_dot', 'L'] = 1.0 / mv
        partials['gam_dot', 'm'] = -(L + T * salpha) / (m * mv)
        partials['gam_dot', 'gam'] = g * sgam / v
        partials['gam_dot', 'alpha'] = T * calpha / mv
        partials['gam_dot', 'v'] = g * cgam / v**2 - (L + T * salpha) / (v * mv)

        partials['h_dot', 'gam'] = v * cgam
        partials['h_dot', 'v'] = sgam

        partials['r_dot', 'gam'] = -v * sgam
        partials['r_dot', 'v'] = cgam


if __name__ == "__main__":

    import openmdao.api as om
    p = om.Problem()
    p.model = FlightPathEOM2D(num_nodes=2)

    p.setup(force_alloc_complex=True)
    p.check_partials(method='cs')
