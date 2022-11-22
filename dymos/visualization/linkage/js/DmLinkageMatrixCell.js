// <<hpp_insert gen/MatrixCell.js>>
// <<hpp_insert js/DmLinkageCellRenderer.js>>
// <<hpp_insert js/DmLinkageSymbolType.js>>

/**
 * A visible cell in the matrix grid.
 * @typedef DmLinkageMatrixCell
 * @property {Number} row Vertical coordinate of the cell in the matrix.
 * @property {Number} col Horizontal coordinate of the cell in the matrix.
 * @property {DmLinkageTreeNode} srcObj The node in the model tree this cell is associated with.
 * @property {DmLinkageTreeNode} tgtObj The model tree node that this outputs to.
 * @property {String} id The srcObj id appended with the tgtObj id.
 * @property {DmLinkageSymbolType} symbolType Info about the type of symbol represented by the node.
 * @property {DmLinkageCellRenderer} renderer The object that draws the cell.
 */
class DmLinkageMatrixCell extends MatrixCell {
    /**
     * Initialize the cell.
     * @param {number} row Vertical coordinate of the cell in the matrix.
     * @param {number} col Horizontal coordinate of the cell in the matrix.
     * @param {DmLinkageTreeNode} srcObj The node in the model tree this node is associated with.
     * @param {DmLinkageTreeNode} tgtObj The model tree node that this outputs to.
     * @param {DmLinkageModelData} model Reference to the model to get some info from it.
     */
    constructor(row, col, srcObj, tgtObj, model) {
        super(row, col, srcObj, tgtObj, model);
    }

    _setSymbolType(model) {
        this.symbolType = new DmLinkageSymbolType(this, model);
    }

    /**
     * Choose a color based on our location and state of the associated OmTreeNode.
     */
    color() {
        const clr = DmLinkageStyle.color;
        if (this.onDiagonal()) {
            if ('warningLevel' in this.obj) {
                const level = this.obj.warningLevel(clr);
                return level.color;
            }
            else {
                if (this.obj.draw.minimized) return clr.collapsed;
                return clr.variableCell;
            }
        }

        return clr.linkageCell;
    }

    /** Choose a renderer based on our SymbolType. */
    _newRenderer() {

        if (! this.inUpperTriangle()) {
            const renderer = super._newRenderer();
            if (renderer) return renderer;
        }

        const color = this.color();

        switch (this.symbolType.name) {
            case "root":
                return new DmLinkageRootCell(color, this.id);
            case "group":
                return new DmLinkageGroupCell(color, this.id);
            case "connected_variable":
            case "connected_parameter":
                return new DmLinkageConnectedCell(color, this.id);
            case "variable":
            case "filter":
                return new DmLinkageVariableCell(color, this.id);
            case "condition":
                return new DmLinkageConditionCell(color, this.id);
            case "phase":
                return new DmLinkagePhaseCell(color, this.id);
            default:
                throw(`No known renderer for ${this.symbolType.name}`);
        }
    }
}
