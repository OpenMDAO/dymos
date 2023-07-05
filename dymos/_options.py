import os

from openmdao.api import OptionsDictionary


options = OptionsDictionary()

_env_check_partials = os.environ.get('DYMOS_CHECK_PARTIALS', '0')
_icp_default = _env_check_partials.lower() in ('1', 'yes', 'true')


def _removed_option(name, value):
    if value is not None:
        raise ValueError(f'Option {name} has been replaced by '
                         'Phase.timeseries_options["use_prefix"].')


options.declare('include_check_partials', default=_icp_default, types=bool,
                desc='If True, include dymos components when checking partials.')

options.declare('plots', default='bokeh', values=['matplotlib', 'bokeh'],
                desc='The plot library used to generate output plots for Dymos.')

options.declare('notebook_mode', default=False, types=bool,
                desc='If True, provide notebook-enhanced plots and outputs.')

options.declare('use_timeseries_prefix', default=None, allow_none=True,
                check_valid=_removed_option,
                desc='Note: This option is no longer valid and has been '
                     'replaced by Phase.timeseries_options["use_prefix"].')
