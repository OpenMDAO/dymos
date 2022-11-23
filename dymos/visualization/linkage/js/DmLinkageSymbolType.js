// <<hpp_insert gen/SymbolType.js>>

/**
 * @typedef {Object} OmSymbolType
 * @property {string} name What the symbol is called.
 * @property {Boolean} declaredPartial Whether the symbol is a declared partial.
 */
class DmLinkageSymbolType extends SymbolType {

    /**
     * Determine the name and whether it's a declared partial based on info
     * from the provided node.
     * @param {MatrixCell} cell The object to select the type from.
     * @param {ModelData} model Reference to the model to get some info from it.
     */
    constructor(cell, model) {
        super(cell, model);
    }

    /** 
     * Decide what object the cell will be drawn as, based on its position
     * in the matrix, type, source, target, and/or other conditions.
     * @param {MatrixCell} cell The cell to operate on.
     * @param {ModelData} model Reference to the entire model.
     */
    getType(cell, model) {
        if (cell.srcObj.isFilter() || cell.tgtObj.isFilter()) {
            this.name = 'filter';
        }
        else if (cell.srcObj.isConnected() && cell.srcObj !== cell.tgtObj ) {
            if (cell.srcObj.isParameter()) {
                this.name = 'connected_parameter';
            }
            else {
                this.name = 'connected_variable';
            }
        }
        else {
            this.name = cell.srcObj.type;
        }
    }
}
