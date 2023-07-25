// <<hpp_insert gen/ModelData.js>>
// <<hpp_insert js/DmLinkageTreeNode.js>>

/** Process the tree, connections, and other info provided about the model. */
class DmLinkageModelData extends ModelData {
    /** Do some discovery in the tree and rearrange & enhance where necessary. */
    constructor(modelJSON) {
        super(modelJSON);
    }

    /**
     * Convert the element to an OmTreeNode. Overrides the superclass method.
     * @param {Object} element The basic properties for the element obtained from JSON.
     * @param {Object} attribNames Map of this model's properties to standard names.
     * @param {DmLinkageTreeNode} parent The node whose children array that this new node will be in.
     * @returns {DmLinkageTreeNode} The newly-created object.
     */
    _newNode(element, attribNames, parent) {
        console.log(element)
        console.log(attribNames)
        return new DmLinkageTreeNode(element, attribNames, parent);
    }
}
