import numpy as np

import openmdao.api as om


class SegmentStateMuxComp(om.ExplicitComponent):
    """
    Class definition for SegmentStateMuxComp.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def initialize(self):
        """
        Declare component options.
        """
        self.options.declare('grid_data', desc='the grid data of the corresponding phase.')
        self.options.declare('state_options', types=dict)

        self.options.declare('output_nodes_per_seg', default=None, types=(int,), allow_none=True,
                             desc='If None, results are provided at the all nodes within each'
                                  'segment.  If an int (n) then results are provided at n '
                                  'equally distributed points in time within each segment.')

    def setup(self):
        """
        Define the independent variables as output variables.
        """
        gd = self.options['grid_data']
        num_seg = gd.num_segments

        if self.options['output_nodes_per_seg'] is None:
            num_nodes = gd.subset_num_nodes['all']
        else:
            num_nodes = num_seg * self.options['output_nodes_per_seg']

        for name, options in self.options['state_options'].items():
            oname = f'states:{name}'

            for i in range(num_seg):
                if self.options['output_nodes_per_seg'] is None:
                    nnps_i = gd.subset_num_nodes_per_segment['all'][i]
                else:
                    nnps_i = self.options['output_nodes_per_seg']

                iname = f'segment_{i}_states:{name}'
                shape = (nnps_i,) + options['shape']

                self.add_input(name=iname, val=np.ones(shape), units=options['units'])

                self.declare_partials(of=oname,  wrt=iname, method='fd')

            self.add_output(name=oname, val=np.ones((num_nodes,) + options['shape']),
                            units=options['units'])

    def compute(self, inputs, outputs):
        """
        Compute component outputs.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        """
        outputs.set_val(inputs.asarray())
