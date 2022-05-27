// <<hpp_insert gen/TreeNode.js>>

/**
 * Extend FilterCapableTreeNode by adding support for feedback cycle arrows and solvers.
 * @typedef DmLinkageTreeNode
 */
class DmLinkageTreeNode extends FilterCapableNode {
    constructor(origNode, attribNames, parent) {
        super(origNode, attribNames, parent);

        if (this.fixed && parent) parent.fixed = true;

        if (this.isPhase()) {
            this.draw.varBoxDims = new Dimensions({'count': 0})
        }

    }

    addFilterChild(attribNames) {
        if (this.isCondition()) { super.addFilterChild(attribNames); }
    }

    isPhase() { return this.type == 'phase'; }

    isCondition() { return this.type == 'condition'; }

    isVariable() { return this.type == 'variable'; }

    isInputOrOutput() { return this.isVariable(); }

    isGroup() { return super.isGroup() || this.isRoot(); }

    isFixed() { return this.fixed; }

    isLinked() { return this.linked; }

    /** In the matrix grid, draw a box around variables that share the same boxAncestor() */
    boxAncestor(level = 2) {
        if (level == 1) { // Return condition reference
            if (this.isVariable()) return this.parent;
            if (this.isCondition()) return this;
        }
        else if (level == 2) { // Return phase reference
            if (this.isVariable()) return this.parent.parent;
            if (this.isCondition()) return this.parent;
        }
        return null;
    }

    /** Not connectable if this is an input group or parents are minimized. */
    isConnectable() {
        if (this.isVariable() && !this.parent.draw.minimized) return true;

        return this.draw.minimized;
    }

}
