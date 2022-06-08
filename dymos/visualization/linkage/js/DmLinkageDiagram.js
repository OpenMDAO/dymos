// <<hpp_insert gen/Diagram.js>>
// <<hpp_insert gen/Layout.js>>
// <<hpp_insert js/DmLinkageStyle.js>>
// <<hpp_insert js/DmLinkageUserInterface.js>>
// <<hpp_insert js/DmLinkageMatrix.js>>

/**
 * Manage all components of the application. The model data, the CSS styles, the
 * user interface, the layout of the matrix, and the matrix grid itself are
 * all member objects. DmLinkageDiagram adds handling for solvers.
 * @typedef DmLinkageDiagram
 */
class DmLinkageDiagram extends Diagram {
    /**
     * Set initial values.
     * @param {Object} modelJSON The decompressed model structure.
     */
    constructor(modelJSON) {
        super(modelJSON);

    }

    _newModelData() {
        this.model = new DmLinkageModelData(this.modelData);
    }

    /** Create a new DmLinkageMatrix object. Overrides superclass method. */
    _newMatrix(lastClickWasLeft, prevCellSize = null) {
        return new DmLinkageMatrix(this.model, this.layout, this.dom.diagGroups,
            this.arrowMgr, lastClickWasLeft, this.ui.findRootOfChangeFunction, prevCellSize);
    }

    /**
     * Separate these calls from the constructor so that subclasses can
     * set values before execution.
     */
     _init() {
        this.style = new DmLinkageStyle(this.dom.svgStyle, this.dims.size.font);
        this.layout = this._newLayout();
        this.ui = new DmLinkageUserInterface(this);
        this.matrix = this._newMatrix(true);
    }
}
