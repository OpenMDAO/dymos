from collections.abc import Sequence

import numpy as np

from .phase import Phase
from ..transcriptions import DirectShooting
from ..transcriptions.grid_data import GridData, GaussLobattoGrid, RadauGrid, UniformGrid
from ..utils.misc import _unspecified


class DirectShootingPhase(Phase):
    """
    The DirectShootingPhase object in Dymos.

    The DirectShootingPhase object in dymos inherits from PhaseBase but is used to override some base methods with ones
    that will warn about certain options or methods being invalid for the DirectShootingPhase.

    Parameters
    ----------
    from_phase : <Phase> or None
        A phase instance from which the initialized phase should copy its data.
    **kwargs : dict
        Dictionary of optional phase arguments.
    """

    def __init__(self, from_phase=None, **kwargs):
        super().__init__(from_phase=from_phase, **kwargs)
        self.simulate_options = None

    def initialize(self):
        """
        Declare instantiation options for the phase.
        """
        super().initialize()
        self.options.declare('method', default='DOP853', desc='The integration method used.')
        self.options.declare('atol', types=float, default=1.0E-6)
        self.options.declare('rtol', types=float, default=1.0E-9)
        self.options.declare('first_step', types=float, allow_none=True, default=None)
        self.options.declare('max_step', types=float, default=np.inf)
        self.options.declare('propagate_derivs', types=bool, default=True,
                             desc='If True, propagate the state and derivatives of the state and time with respect to '
                                  'the integration parameters. If False, only propagate the primal states. If only '
                                  'using this transcription to propagate an ODE and derivatives are nor needed, '
                                  'setting this option to False should result in faster execution.')
        self.options.declare('subprob_reports', default=False,
                             desc='Controls the reports made when running the subproblems for DirectShooting')
        self.options.declare('input_grid', types=(GaussLobattoGrid, RadauGrid),
                             desc='Control values are defined at the control_input nodes of the input grid.')
        self.options.declare('output_grid', types=(GaussLobattoGrid, RadauGrid, UniformGrid), allow_none=True,
                             default=None,
                             desc='The grid distribution determining the location of the output nodes. The default '
                                  'value of None will result in the use of the input_grid for outputs. This is useful '
                                  'for the implementation of path constraints but can result in highly nonlinear '
                                  'dynamics being smoothed over in the outputs. When used for validation through '
                                  'simulation, it is generally wise to choose an output grid that is more dense '
                                  'than the input grid to capture this nonlinearity.')

    def setup(self):
        """
        Build the model hierarchy for a Dymos DirectShootingPhase.
        """
        # Finalize the variables if it hasn't happened already.
        # If this phase exists within a Trajectory, the trajectory will finalize them during setup.
        tx = self.options['transcription'] = DirectShooting(method=self.options['method'],
                                                            atol=self.options['atol'],
                                                            rtol=self.options['rtol'],
                                                            first_step=self.options['first_step'],
                                                            max_step=self.options['max_step'],
                                                            propagate_derivs=self.options['propagate_derivs'],
                                                            subprob_reports=self.options['subprob_reports'],
                                                            input_grid=self.options['input_grid'],
                                                            output_grid=self.options['output_grid'])
        tx.setup_time(self)

        if self.control_options:
            tx.setup_controls(self)

        if self.polynomial_control_options:
            tx.setup_polynomial_controls(self)

        if self.parameter_options:
            tx.setup_parameters(self)

        tx.setup_states(self)
        self._check_ode()
        tx.setup_ode(self)

        tx.setup_timeseries_outputs(self)
        tx.setup_defects(self)
        tx.setup_solvers(self)

    def simulate(self, times_per_seg=10, method=_unspecified, atol=_unspecified, rtol=_unspecified,
                 first_step=_unspecified, max_step=_unspecified, record_file=None):
        """
        Stub to make sure users are informed that simulate cannot be done on DirectShootingPhase.

        Parameters
        ----------
        times_per_seg : int or None
            Number of equally spaced times per segment at which output is requested.  If None,
            output will be provided at all Nodes.
        method : str
            The scipy.integrate.solve_ivp integration method.
        atol : float
            Absolute convergence tolerance for scipy.integrate.solve_ivp.
        rtol : float
            Relative convergence tolerance for scipy.integrate.solve_ivp.
        first_step : float
            Initial step size for the integration.
        max_step : float
            Maximum step size for the integration.
        record_file : str or None
            If a string, the file to which the result of the simulation will be saved.
            If None, no record of the simulation will be saved.

        Returns
        -------
        problem
            An OpenMDAO Problem in which the simulation is implemented.  This Problem interface
            can be interrogated to obtain timeseries outputs in the same manner as other Phases
            to obtain results at the requested times.
        """
        raise NotImplementedError('Method `simulate` is not available for DirectShootingPhase.')

    def set_simulate_options(self, method=_unspecified, atol=_unspecified, rtol=_unspecified,
                             first_step=_unspecified, max_step=_unspecified):
        """
        Stub to make sure users are informed that simulate cannot be done on DirectShootingPhase.

        Parameters
        ----------
        method : str
            The scipy.integrate.solve_ivp integration method.
        atol : float
            Absolute convergence tolerance for scipy.integrate.solve_ivp.
        rtol : float
            Relative convergence tolerance for scipy.integrate.solve_ivp.
        first_step : float
            Initial step size for the integration.
        max_step : float
            Maximum step size for the integration.

        Returns
        -------
        problem
            An OpenMDAO Problem in which the simulation is implemented.  This Problem interface
            can be interrogated to obtain timeseries outputs in the same manner as other Phases
            to obtain results at the requested times.
        """
        raise NotImplementedError('Method set_simulate_options is not available for DirectShootingPhase.')
