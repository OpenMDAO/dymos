// <<hpp_insert gen/CellRenderer.js>>

class DmLinkageConditionCell extends GroupBase {
    constructor(color, id) {
        super(color, id);
    }
}

class DmLinkagePhaseCell extends GroupBase {
    constructor(color, id) {
        super(color, id);
    }
}

class DmLinkageConnectedCell extends VectorBase {
    constructor(color, id) {
        super(color, id);
    }

    /**
     * Select the element with D3 if not already done, attach a transition
     * and resize the shape.
     * @param svgGroup Reference to SVG <g> element associated with data.
     * @param {Object} dims The cell spec to use while resizing/repositioning.
     */
     update(svgGroup, dims) {
        const d3Group = d3.select(svgGroup);

        d3Group.select('rect').transition(sharedTransition)
            .attr("x", dims.topLeft.x * dims.size.percent)
            .attr("y", dims.topLeft.y * dims.size.percent)
            .attr("width", dims.bottomRight.x * dims.size.percent * 2)
            .attr("height", dims.bottomRight.y * dims.size.percent * 2);

        d3Group.select('text').transition(sharedTransition)
            .attr("x", 0)
            .attr("y", 0)
            .style('font-size', `${dims.bottomRight.x * dims.size.percent * 2}px`);
        

        return d3Group.selectAll('*');
    }

    /** 
     * Get the D3 selection for the appropriate group and append a filled rectangle.
     * @param {Object} svgGroup Reference to SVG <g> element associated with data.
     * @param {Object} dims The cell spec to use while rendering.
     */
     render(svgGroup, dims) {
        const d3Group = d3.select(svgGroup);

        d3Group
            .append('rect')
            .attr("class", this.className)
            .attr("id", this.id)
            .style("fill", this.color);

        d3Group
            .append('text')
            .attr('class', 'connected-variable-text')
            .attr('id', `${this.id}-text`)
            .html('&#x2794;');

        return this.update(svgGroup, dims);
    }
}

class DmLinkageVariableCell extends VectorBase {
    constructor(color, id) {
        super(color, id);
    }
}

class DmLinkageCell extends VectorBase {
    constructor(color, id) {
        super(color, id);
    }
}

class DmLinkageGroupCell extends GroupBase {
    constructor(color, id) {
        super(color, id);
    }
}

class DmLinkageRootCell extends GroupBase {
    constructor(color, id) {
        super(color, id);
    }
}
