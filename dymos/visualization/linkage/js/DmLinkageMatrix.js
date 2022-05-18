// <<hpp_insert js/DmLinkageMatrixCell.js>>
// <<hpp_insert gen/Matrix.js>>

/**
 * Use the model tree to build the matrix of variables and connections, display, and
 * perform operations with it.
 * @typedef DmLinkageMatrix
 * @property {DmLinkageTreeNodes[]} nodes Reference to nodes that will be drawn.
 * @property {DmLinkageModelData} model Reference to the pre-processed model.
 * @property {Layout} layout Reference to object managing columns widths and such.
 * @property {Object} diagGroups References to <g> SVG elements created by Diagram.
 * @property {Number} levelOfDetailThreshold Don't draw elements below this size in pixels.
 * @property {Object} nodeSize Width and height of each node in the matrix.
 * @property {Object} prevNodeSize Width and height of each node in the previous matrix.
 * @property {Object[][]} grid Object keys corresponding to rows and columns.
 * @property {DmLinkageMatrixCell[]} visibleCells One-dimensional array of all cells, for D3 processing.
 */
class DmLinkageMatrix extends Matrix {
    /**
     * Render the matrix of visible elements in the model.
     * @param {DmLinkageModelData} model The pre-processed model data.
     * @param {Layout} layout Pre-computed layout of the diagram.
     * @param {Object} diagGroups References to <g> SVG elements created by Diagram.
     * @param {ArrowManager} arrowMgr Object to create and manage conn. arrows.
     * @param {Boolean} lastClickWasLeft
     * @param {function} findRootOfChangeFunction
     */
    constructor(model, layout, diagGroups, arrowMgr, lastClickWasLeft, findRootOfChangeFunction,
        prevNodeSize = { 'width': 0, 'height': 0 }) {
        super(model, layout, diagGroups, arrowMgr, lastClickWasLeft, findRootOfChangeFunction, prevNodeSize);
    }

    /**
     * Generate a new DmLinkageMatrixCell object. Overrides superclass definition.
     * @param {Number} row Vertical coordinate of the cell in the matrix.
     * @param {Number} col Horizontal coordinate of the cell in the matrix.
     * @param {DmLinkageTreeNode} srcObj The node in the model tree this node is associated with.
     * @param {DmLinkageTreeNode} tgtObj The model tree node that this outputs to.
     * @param {ModelData} model Reference to the model to get some info from it.
     * @returns {DmLinkageMatrixCell} Newly created cell.
     */
    _createCell(row, col, srcObj, tgtObj, model) {
        return new DmLinkageMatrixCell(row, col, srcObj, tgtObj, model);
    }


}
