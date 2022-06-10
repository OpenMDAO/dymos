// <<hpp_insert gen/Layout.js>>

/**
 * Calculates and stores the size and positions of visible elements.
 * @typedef DmLinkageLayout
 * @property {ModelData} model Reference to the preprocessed model.
 * @property {DmLinkageTreeNode} zoomedElement Reference to zoomedElement managed by Diagram.
 * @property {DmLinkageTreeNode[]} zoomedNodes  Child workNodes of the current zoomed element.
 * @property {DmLinkageTreeNode[]} visibleNodes Zoomed workNodes that are actually drawn.
 * @property {DmLinkageTreeNode[]} zoomedSolverNodes Child solver workNodes of the current zoomed element.
 * @property {Object} svg Reference to the top-level SVG element in the document.
 * @property {Object} size The dimensions of the model and solver trees.
 */
class DmLinkageLayout extends Layout {
    /**
     * Compute the new layout based on the model data and the zoomed element.
     * @param {ModelData} model The pre-processed model object.
     * @param {Object} newZoomedElement The element the new layout is based around.
     * @param {Object} dims The initial sizes for multiple tree elements.
     */
    constructor(model, zoomedElement, dims) {
        super(model, zoomedElement, dims, false);

        this._init();
    }

    /** Set up the solver tree layout. */
    _init() {
        super._init();

        return this;
    }

    /**
     * Determine the text associated with the node. Normally its name,
     * but may be a filter.
     * @param {DmLinkageTreeNode} node The item to operate on.
     * @return {String} The selected text.
     */
     getText(node) {
        return node.isFilter()? 'Filtered Variables' : node.name;
    }
}
