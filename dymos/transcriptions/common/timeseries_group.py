import openmdao.api as om

from .timeseries_output_comp import TimeseriesOutputComp


class TimeseriesOutputGroup(om.Group):
    """
    Class definition of the TimeComp.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize(self):
        """
        Declare component options.
        """
        self.options.declare('timeseries_output_comp', types=TimeseriesOutputComp, recordable=False,
                             desc='Timeseries component specific to the transcription of the optimal control problem.')

        self.options.declare('has_expr', types=bool,
                             desc='If true, timeseries group has an expression to be computed')

    def setup(self):
        """
        Define the structure of the timeseries group.
        """
        timeseries_output_comp = self.options['timeseries_output_comp']
        has_expr = self.options['has_expr']
        if has_expr:
            self.add_subsystem('timeseries_exec_comp', om.ExecComp(has_diag_partials=True))

        self.add_subsystem('timeseries_comp', timeseries_output_comp,
                           promotes_inputs=['*'], promotes_outputs=['*'])
