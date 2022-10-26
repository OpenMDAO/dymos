import openmdao.api as om
import numpy as np

from .timeseries_output_comp import TimeseriesOutputCompBase


class TimeseriesOutputGroup(om.Group):
    def initialize(self):
        self.options.declare('timeseries_output_comp', types=TimeseriesOutputCompBase,
                             desc='Timeseries component specific to the transcription of the optimal control problem.')

        self.options.declare('has_expr', types=bool,
                             desc='If true, timeseries group has an expression to be computed')

    def setup(self):
        timeseries_output_comp = self.options['timeseries_output_comp']
        has_expr = self.options['has_expr']
        if has_expr:
            self.add_subsystem('timeseries_exec_comp', om.ExecComp(has_diag_partials=True))

        self.add_subsystem('timeseries_comp', timeseries_output_comp,
                           promotes_inputs=['*'], promotes_outputs=['*'])
