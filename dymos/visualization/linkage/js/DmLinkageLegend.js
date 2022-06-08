// <<hpp_insert gen/Legend.js>>

/**
 * Draw a symbol describing each of the element types.
 * @typedef DmLinkageLegend
 * @property {Boolean} shown Whether the legend is currently drawn or not.
 */
class DmLinkageLegend extends Legend {
    /**
     * Initializes the legend object.
     * @param {ModelData} modelData In the base class, symbols only appear if they're in the model
     */
    constructor(modelData) {
        super(modelData);
    }

    /** Define all the legend items, colors, and styles. */
    _initItemTypes() {

        this.nodeTypes = [
            {
                'name': 'Phase',
                'color': DmLinkageStyle.color.phase
            },
            {
                'name': 'Initial Condition/Variable',
                'color': DmLinkageStyle.color.initial
            },
            {
                'name': 'Final Condition/Variable',
                'color': DmLinkageStyle.color.final
            },
            {
                'name': 'Collapsed',
                'color': DmLinkageStyle.color.collapsed
            }
        ]

         this.cellTypes = [
            {
                'name': 'Variable',
                'color': DmLinkageStyle.color.variableCell,
                'cssClass': 'dm-legend-box'
            },
            {
                'name': 'Fixed Variable',
                'color': DmLinkageStyle.color.fixedUnlinkedVariableCell,
                'cssClass': 'dm-legend-box'
            },
            {
                'name': 'Fixed, Linked Variable',
                'color': DmLinkageStyle.color.fixedLinkedVariableCell,
                'cssClass': 'dm-legend-box'
            },
            {
                'name': 'Linkage',
                'color': DmLinkageStyle.color.linkageCell,
                'cssClass': 'dm-legend-cell'
            },
            {
                'name': 'Fixed Linkage',
                'color': DmLinkageStyle.color.fixedLinkageCell,
                'cssClass': 'dm-legend-cell'
            },
            {
                'name': 'Connected',
                'color': 'none',
                'cssClass': 'dm-legend-connected'
            }
        ];
    }

    _setDisplayBooleans() { }

    /** Add symbols for all of the items that were defined */
    _setupContents() {
        const nodeLegend = d3.select('#tree-nodes-legend');
        for (const nodeType of this.nodeTypes) {
            this._addItem(nodeType, nodeLegend);
        }

        nodeLegend.style('width', nodeLegend.node().scrollWidth + 'px')

        const cellLegend = d3.select('#matrix-cells-legend');
        for (const cellType of this.cellTypes) {
            this._addItem(cellType, cellLegend, cellType.cssClass);
        }

        cellLegend.style('width', cellLegend.node().scrollWidth + 'px')

    }

}
