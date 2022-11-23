// <<hpp_insert gen/TreeNode.js>>

/**
 * Extend FilterCapableTreeNode by adding support for feedback cycle arrows and solvers.
 * @typedef DmLinkageTreeNode
 */
class DmLinkageTreeNode extends FilterCapableNode {
    constructor(origNode, attribNames, parent) {
        super(origNode, attribNames, parent);

        if ((this.isPhase() || this.isCondition()) && this.name == 'params' ) {
            this.draw.minimized = true;
        }

        if (parent) { // Retain "highest warning" cell color when collapsed
            // Only applies to trajectory parameters:
            if (this.paramOpt == false) parent.paramOpt = false;
        }

        if (this.isPhase()) {
            this.draw.varBoxDims = new Dimensions({'count': 0})
        }

    }

    /** Temp fix to gen code reference */
    get absPathName() { return this.path; }

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

    isTrajectoryParameter() {
        return (this.isParameter() && this.parent.parent.isTrajectory()) ||
            (this.name == 'params' && this.parent.isTrajectory()); // For the collapsed cell
    }

    isInputOrOutput() { return this.isVariable(); }

    isGroup() { return super.isGroup() || this.isRoot(); }

    isFixed() { return this.fixed; }

    isLinked() { return this.linked; }

    /**
     * Determine the highest warning level for the node itself, or
     * find the highest warning level of its children.
     * @param {Object} colors A list of color definitions.
     * @returns {Object} Property color is the chosen color, priority is used within
     *                   this recursive function to help with selection.
     */
    warningLevel(colors) {
        let clr = colors.variableCell;
        let priority = 0;

        if (this.isVariable()) {
            if (this.isTrajectoryParameter() && this.paramOpt === false) {
                clr = colors.falseParamOpt;
                priority = 1;
            }
            else if (this.isParameter()) {
                if (this.isFixed()) {
                    if (this.isLinked()) {
                        clr = colors.fixedLinkedVariableCell;
                        priority = 2;
                    }
                }
            }
            else if (this.isFixed()) {
                if (this.isLinked()) {
                    clr = colors.fixedLinkedVariableCell;
                    priority = 2;
                }
                else {
                    clr = colors.fixedUnlinkedVariableCell;
                    priority = 1;
                }
            }
        }
        else if (this.hasChildren()) { // Must be a collapsed parent
            clr = colors.collapsed;
            for (const child of this.children) {
                if ('warningLevel' in child) {
                    const level = child.warningLevel(colors);
                    if (level.priority > priority) {
                        clr = level.color;
                        priority = level.priority;
                    }
                }
            }
        }

        return { 'color': clr, 'priority': priority };
    }

    /**
     * Determine if this node or any children are connected.
     * @returns True if a connection is found.
     */
    isConnected() {
        if (this.isVariable()) return this.connected;

        if (this.hasChildren()) {
            for (const child of this.children) {
                if ('isConnected' in child && child.isConnected()) return true;
            }
        }

        return false;
    }

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
