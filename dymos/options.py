from openmdao.api import OptionsDictionary

options = OptionsDictionary()

options.declare('include_check_partials', default=True, types=bool,
                desc='If True, include dymos components when checking partials.')

options.declare('plots', default='matplotlib', values=['matplotlib', 'bokeh'],
                desc='The plot library used to generate output plots for Dymos.')

options.declare('notebook_mode', default=False, types=bool,
                desc='If True, provide notebook-enhanced plots and outputs.')
