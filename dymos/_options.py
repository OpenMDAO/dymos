from openmdao.api import OptionsDictionary


options = OptionsDictionary()


def _removed_option(name, value):
    if value is not None:
        raise ValueError(f'Option {name} has been replaced by '
                         'Phase.timeseries_options["use_prefix"].')


options.declare('plots', default='bokeh', values=['matplotlib', 'bokeh'],
                desc='The plot library used to generate output plots for Dymos.')

options.declare('notebook_mode', default=False, types=bool,
                desc='If True, provide notebook-enhanced plots and outputs.')

options.declare('use_timeseries_prefix', default=None, allow_none=True,
                check_valid=_removed_option,
                desc='Note: This option is no longer valid and has been '
                     'replaced by Phase.timeseries_options["use_prefix"].')
