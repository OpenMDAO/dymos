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

    /** Use a single filter instead of separate inputs and outputs */
    addFilterChild(attribNames) {
        if (this.isCondition()) {
            this.filter = new FilterNode(this, attribNames, 'variables');
            this.children.push(this.filter);
        }
    }

    /** Add ourselves to the parental filter */
    addSelfToFilter() { this.parent.filter.add(this);  }

    /** Remove ourselves from the parental filter */
    removeSelfFromFilter() { this.parent.filter.del(this); }

    getFilterList() { return [ this.filter ];}

    wipeFilters() { this.filter.wipe(); }

    addToFilter(node) { this.filter.add(node); }

    isPhase() { return this.type == 'phase'; }

    isCondition() { return this.type == 'condition'; }

    isVariable() { return this.type == 'variable'; }

    isTrajectory() { return this.type == 'root'; }

    isParameter() { return this.isVariable() && this.class == 'parameter'; }

    isTrajectoryParameter() { return this.isParameter() && this.parent.parent.isTrajectory(); }

    isInputOrOutput() { return this.isVariable(); }

    isGroup() { return super.isGroup() || this.isRoot(); }

    isFixed() { return this.fixed; }

    isLinked() { return this.linked; }

    isConnected() { return this.isVariable() && this.connected == true; }

    /** In the matrix grid, draw a box around variables that share the same boxAncestor() */
    boxAncestor(level = 2) {
        if (level == 1) { // Return condition reference
            if (this.isVariable()) return this.parent;
            if (this.isCondition()) return this;
        }
        else if (level == 2) { // Return phase reference
            if (this.isVariable() && this.parent.isPhase()) return this.parent;
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
