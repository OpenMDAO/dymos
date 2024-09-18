import numpy as np
import openmdao.api as om

from ...utils.misc import get_rate_units
from ...utils.lgl import lgl

from scipy.interpolate import CubicSpline


class CubicSplineControlInterpComp(om.ExplicitComponent):
    """
    A component which interpolates control values in 1D using cubic spline interpolation.

    Takes training values for control variables at given _input_ nodes,
    broadcasts them to _discretization_ nodes, and then interpolates the discretization values
    to provide a control variable at a given segment tau or phase tau.

    This interpolation method is intended for use in simulation phases and should not be used to
    solve optimization problems.

    For dynamic controls, the current segment is given as a discrete input and the interpolation is
    a smooth polynomial along the given segment.

    OpenMDAO assumes sizes of variables at setup time, and we don't want to need to change the
    size of the control input nodes when we evaluate different segments. Instead, this component
    will take in the control values of all segments and internally use the appropriate one.

    Parameters
    ----------
    grid_data : GridData
        A GridData instance that details information on how the control input and discretization
        nodes are layed out.
    control_options : dict of {str: ControlOptionsDictionary}
        A mapping that maps the name of each control to a ControlOptionsDictionary of its options.
    time_units : str
        The time units pertaining to the control rates.
    standalone_mode : bool
        If True, this component runs its configuration steps during setup. This is useful for
        unittests in which the component does not exist in a larger group.
    **kwargs
        Keyword arguments passed to ExplicitComponent.
    """
    def __init__(self, grid_data, control_options=None,
                 time_units=None, standalone_mode=False, **kwargs):
        self._grid_data = grid_data
        self._control_options = {} if control_options is None else control_options
        self._time_units = time_units
        self._standalone_mode = standalone_mode

        # Cache formatted strings: { control_name : (input_name, output_name) }
        self._control_io_names = {}
        self._polynomial_control_nodes = {}
        self.num_uhat_nodes = None
        self._input_grid = []

        super().__init__(**kwargs)

    def initialize(self):
        """
        Declare component options.
        """
        self.options.declare('segment_index', types=int, desc='index of the current segment')
        self.options.declare('vec_size', types=int, default=1,
                             desc='number of points at which the control will be evaluated. This is not'
                                  'necessarily the same as the number of nodes in the GridData.')
        self.options.declare('compute_derivs', types=bool, default=True,
                             desc='Set to True if the interpolant needs to also compute the derivatives '
                             'of the outputs, otherwise False. This should be set to False for simulation mode '
                             'and true when using ExplicitShooting for optimization')

    def _configure_controls(self):
        vec_size = self.options['vec_size']
        gd = self._grid_data

        if not self._control_options:
            return

        self.num_uhat_nodes = gd.subset_num_nodes['control_input']
        for control_name, options in self._control_options.items():
            if options['control_type'] == 'full':
                shape = options['shape']
                units = options['units']
                input_name = f'controls:{control_name}'
                output_name = f'control_values:{control_name}'
                rate_name = f'control_rates:{control_name}_rate'
                rate2_name = f'control_rates:{control_name}_rate2'
                rate_units = get_rate_units(units, self._time_units)
                rate2_units = get_rate_units(units, self._time_units, deriv=2)
                uhat_shape = (self.num_uhat_nodes,) + shape
                output_shape = (vec_size,) + shape
                self.add_input(input_name, shape=uhat_shape, units=units)
                self.add_output(output_name, shape=output_shape, units=units)
                self.add_output(rate_name, shape=output_shape, units=rate_units)
                self.add_output(rate2_name, shape=output_shape, units=rate2_units)
                self._control_io_names[control_name] = (input_name, output_name, rate_name, rate2_name)

                input_ptau_grid = self._grid_data.node_ptau[self._grid_data.subset_node_indices['control_input']]
                _, self._input_grid = np.unique(input_ptau_grid, return_index=True)

            else:
                order = options['order']
                shape = options['shape']
                units = options['units']
                input_name = f'controls:{control_name}'
                output_name = f'control_values:{control_name}'
                rate_name = f'control_rates:{control_name}_rate'
                rate2_name = f'control_rates:{control_name}_rate2'
                rate_units = get_rate_units(units, self._time_units)
                rate2_units = get_rate_units(units, self._time_units, deriv=2)
                input_shape = (order + 1,) + shape
                output_shape = (vec_size,) + shape
                self.add_input(input_name, shape=input_shape, units=units)
                self.add_output(output_name, shape=output_shape, units=units)
                self.add_output(rate_name, shape=output_shape, units=rate_units)
                self.add_output(rate2_name, shape=output_shape, units=rate2_units)
                self._control_io_names[control_name] = (input_name, output_name, rate_name, rate2_name)
                self._polynomial_control_nodes[control_name], _ = lgl(order+1)

    def setup(self):
        """
        Perform the I/O creation if operating in _standalone_mode.
        """
        if self._standalone_mode:
            self.configure_io()

    def set_segment_index(self, idx, **kwargs):
        """
        Set the active segment index for control interpolation.

        Parameters
        ----------
        idx : int
            The index of the segment in which the controls are to be interpolated.
        **kwargs : dict, optional
            Keyword arguments that make this interpolant call-compatible with the BarycentricLagrangeInterpolant.
        """
        self.options['segment_index'] = idx

    def configure_io(self):
        """
        I/O creation is delayed until configure so we can determine shape and units for the controls.
        """
        vec_size = self.options['vec_size']

        self.add_input('stau', shape=(vec_size,), units=None)
        self.add_input('dstau_dt', val=1.0, units=f'1/{self._time_units}')
        self.add_input('t_duration', val=1.0, units=self._time_units)
        self.add_input('ptau', shape=(vec_size,), units=None)

        self._configure_controls()

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Compute interpolated control values and rates.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        discrete_inputs : `Vector`
            `Vector` containing discrete_inputs.
        discrete_outputs : `Vector`
            `Vector` containing discrete_outputs.
        """
        vec_size = self.options['vec_size']
        ptau = inputs['ptau']
        ptau_grid = self._grid_data.node_ptau

        if self._control_options:
            input_node_idxs = self._grid_data.subset_node_indices['control_input']

            for control_name, options in self._control_options.items():
                if options['control_type'] == 'full':
                    input_name, output_name, rate_name, rate2_name = self._control_io_names[control_name]
                    shape = options['shape']

                    size = np.prod(shape)
                    out = np.zeros((vec_size, size))
                    rate = np.zeros((vec_size, size))
                    rate2 = np.zeros((vec_size, size))

                    for i in range(size):
                        spl = CubicSpline(ptau_grid[input_node_idxs][self._input_grid],
                                          inputs[input_name][self._input_grid].flatten('F')[self.num_uhat_nodes*i:
                                                                                            self.num_uhat_nodes*(i+1)])
                        out[:, i] = spl(ptau)
                        rate[:, i] = spl(ptau, nu=1) / (0.5 * inputs['t_duration'])
                        rate2[:, i] = spl(ptau, nu=2) / (0.5 * inputs['t_duration'])**2

                    outputs[output_name] = out.reshape((vec_size,) + shape, order='F')
                    outputs[rate_name] = rate.reshape((vec_size,) + shape, order='F')
                    outputs[rate2_name] = rate2.reshape((vec_size,) + shape, order='F')
                else:
                    input_name, output_name, rate_name, rate2_name = self._control_io_names[control_name]
                    order = options['order']
                    poly = np.polyfit(self._polynomial_control_nodes[control_name], inputs[input_name].ravel(), order)
                    der1 = np.polyder(poly)
                    der2 = np.polyder(der1)
                    outputs[output_name] = np.polyval(poly, ptau)
                    outputs[rate_name] = np.polyval(der1, ptau) / (0.5 * inputs['t_duration'])
                    outputs[rate2_name] = np.polyval(der2, ptau) / (0.5 * inputs['t_duration'])**2
