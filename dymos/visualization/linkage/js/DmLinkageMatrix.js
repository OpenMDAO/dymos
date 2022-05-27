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

    _init() {
        CellRenderer.updateDims(this.nodeSize.width, this.nodeSize.height, 0.8);
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

    /**
     * Create an array containing the dimensions of the background "checkerboard" pattern
     * for the matrix, with each square based on the presence of a condition.
     * @param {Dimensions} cellDims The size of a matrix cell.
     */
    _plotBgSquares(cellDims) {
        const boxInfo = this._boxInfo(1);

        const conditionBoxInfo = [];
        for (let i = 0; i < boxInfo.length; ++i) {
            const box = boxInfo[i];

            const curNode = this.diagNodes[box.startI];
            if (!curNode.boxAncestor(1)) continue; // Collapsed phase

            i = box.stopI;

            // Save dimensions to the underlying tree node so we can properly
            // transition when the diagram changes.
            if (! curNode.draw.bgBoxDims) {
                curNode.draw.bgBoxDims = new Dimensions({ 'start': 0, 'stop': 0})
            }
            curNode.draw.bgBoxDims.preserve().set({ 'start': box.startI, 'stop': box.stopI });

            conditionBoxInfo.push(box);
        }        

        this._bgSquareDims = [];

        for (let boxYidx in conditionBoxInfo) {
            const boxY = conditionBoxInfo[boxYidx];
            const nodeY = this.diagNodes[boxY.startI];
            const prevY = nodeY.draw.bgBoxDims.prev;

            for (let boxXidx in conditionBoxInfo) {
                const boxX = conditionBoxInfo[boxXidx];
                const nodeX = this.diagNodes[boxX.startI];
                const prevX = nodeX.draw.bgBoxDims.prev;

                let boxType = null;

                const evenX = (boxXidx % 2 == 0), evenY = (boxYidx % 2 == 0);
                if (evenX && evenY) boxType = 0;
                if (!evenX && evenY) boxType = 1;
                if (evenX && !evenY) boxType = 2;
                if (!evenX && !evenY) boxType = 3;

                const bgSquare = {
                    'id': `bg-box-${boxType}-${nodeX.boxAncestor(1).path}-${nodeY.boxAncestor(1).path}`,
                    'type': boxType,
                    'dims': new Dimensions({
                        'x': cellDims.width * boxX.startI,
                        'y': cellDims.height * boxY.startI,
                        'width': cellDims.width * (boxX.stopI - boxX.startI + 1),
                        'height': cellDims.height * (boxY.stopI - boxY.startI + 1)
                    }, null,
                    {
                        'x': cellDims.prev.width * prevX.start,
                        'y': cellDims.prev.height * prevY.start,
                        'width': cellDims.prev.width * (prevX.stop - prevX.start + 1),
                        'height': cellDims.prev.height * (prevY.stop - prevY.start + 1)                    
                    })
                }
                this._bgSquareDims.push(bgSquare)
            }
        }
    }

    /**
     * Draw the "checkerboard" pattern of background squares on the matrix diagram,
     * each square related to a condition.
     * @param {Dimensions} cellDims The size of a matrix cell.
     */
    _drawBgSquares(cellDims) {
        const self = this;
        this._plotBgSquares(cellDims);

        this.diagGroups.background.selectAll('g.condition-box')
            .data(this._bgSquareDims, d => d.id)
            .join(
                enter => {
                    const newGroups = enter.append('g')
                        .attr('class', 'condition-box')
                        .attr('transform', d => `translate(${d.dims.prev.x}, ${d.dims.prev.y})`);

                    newGroups.transition(sharedTransition)
                        .attr('transform', d => `translate(${d.dims.x}, ${d.dims.y})`);

                    newGroups.append('rect')
                        .attr('class', d => `cond-box-${d.type}`)
                        .attr('width', d => d.dims.prev.width)
                        .attr('height', d => d.dims.prev.height)
                        .transition(sharedTransition)
                        .attr('width', d => d.dims.width)
                        .attr('height', d => d.dims.height);    
                },
                update => {
                    update.transition(sharedTransition)
                        .attr('transform', d =>
                            `translate(${d.dims.x}, ${d.dims.y})`);

                    update.select('rect').transition(sharedTransition)
                        .attr('width', d => d.dims.width)
                        .attr('height', d => d.dims.height);
                },
                exit => {
                    const ratioX = cellDims.width / cellDims.prev.width;
                    const ratioY = cellDims.height / cellDims.prev.height;

                    exit.transition(sharedTransition)
                        .attr('transform', d =>
                            `translate(${d.dims.x * ratioX}, ${d.dims.y * ratioY})`)
                        .remove();
                    
                    exit.select('rect').transition(sharedTransition)
                        .attr('width', d => d.dims.width * ratioX)
                        .attr('height', d => d.dims.height * ratioY)
                        .remove();
                }
            )
    }

    /**
     * Compute the coordinates of each line in the grid that identifies variables with a phase.
     * @param {String} phasePath The path of the associated phase, used as an identifier.
     * @param {Dimensions} cellDims The size of the current and previous matrix cell.
     * @param {Dimensions} boxDims The current and previous counts of variables in the box.
     * @returns {Object[]} An array of objects each w/an id property and a dims property.
     */
    _plotBoxGrid(phasePath, cellDims, boxDims) {
        const prevBoxWidth = cellDims.prev.width * boxDims.prev.count,
              prevBoxHeight = cellDims.prev.height * boxDims.prev.count,
              boxWidth = cellDims.width * boxDims.count,
              boxHeight = cellDims.height * boxDims.count;

        const gridLines = [];

        // Vertical lines
        for (let x = 0; x <= boxDims.count; x++) {
            gridLines.push({
                'id': `gridbox-${phasePath}-vert-${x}`,
                'dims': new Dimensions({
                    'x1': x * cellDims.width,
                    'y1': 0,
                    'x2': x * cellDims.width,
                    'y2': boxHeight,
                },
                null,
                {
                    'x1': prevBoxWidth,
                    'y1': 0,
                    'x2': prevBoxWidth,
                    'y2': prevBoxHeight, 
                })
            })
        }

        for (let y = 0; y <= boxDims.count; y++) {
            gridLines.push({
                'id': `gridbox-${phasePath}-horiz-${y}`,
                'dims': new Dimensions({
                    'x1': 0,
                    'y1': y * cellDims.height,
                    'x2': boxWidth,
                    'y2': y * cellDims.height,
                },
                null,
                {
                    'x1': 0,
                    'y1': prevBoxHeight,
                    'x2': prevBoxWidth,
                    'y2': prevBoxHeight
                })
            })
        }

        return gridLines;
    }

    /**
     * Draw gridlines around the variables found in a single phase.
     * @param {Object} boxGrpNode The D3 <g> selection to add lines to.
     * @param {String} phasePath The path of the phase to use as an identifier.
     * @param {Dimensions} dims The current and previous matrix cell dimensions.
     * @param {Dimensions} boxDims The current and previous count of variables displayed in the box.
     * @returns The same boxGrpNode that was used as an argument.
     */
    _drawBox(boxGrpNode, phasePath, dims, boxDims) {
        const gridLines = this._plotBoxGrid(phasePath, dims, boxDims);

        boxGrpNode.selectAll('line.boxgrid')
            .data(gridLines, d => d.id)
            .join(
                enter => {
                    enter.append('line')
                        .attr('class', 'boxgrid')
                        .attr('x1', d => d.dims.prev.x1).attr('y1', d => d.dims.prev.y1)
                        .attr('x2', d => d.dims.prev.x2).attr('y2', d => d.dims.prev.y2)
                        .transition(sharedTransition)
                        .attr('x1', d => d.dims.x1).attr('y1', d => d.dims.y1)
                        .attr('x2', d => d.dims.x2).attr('y2', d => d.dims.y2)
                },
                update => {
                    update.transition(sharedTransition)
                        .attr('x1', d => d.dims.x1).attr('y1', d => d.dims.y1)
                        .attr('x2', d => d.dims.x2).attr('y2', d => d.dims.y2)
                },
                exit => {
                    exit.transition(sharedTransition)
                        .attr('x1', function() { return d3.select(this).attr('x1') == 0? 0 : dims.width * boxDims.count; })
                        .attr('y1', function() { return d3.select(this).attr('y1') == 0? 0 : dims.height * boxDims.count; })
                        .attr('x2', dims.width * boxDims.count)
                        .attr('y2', dims.height * boxDims.count)
                        .remove();
                    }

            );

        return boxGrpNode;
    }

    /** Draw boxes around the cells associated with each variable grouping. */
    _drawVariableBoxes(dims) {
        const self = this; 

        self.diagGroups.variableBoxes.selectAll('g.variable_box')
            .data(self._variableBoxInfo, d => d.obj.id)
            .join(
                enter => {
                    const newGroups = enter.append('g')
                        .attr('class', 'variable_box')
                        .attr('transform', d => {
                            const transX = dims.prev.width * (d.startI - enterIndex),
                                transY = dims.prev.height * (d.startI - enterIndex);
                            return `translate(${transX}, ${transY})`;
                        });

                    newGroups.transition(sharedTransition)
                        .attr('transform', d =>
                            `translate(${dims.width * d.startI}, ${dims.height * d.startI})`);

                    newGroups.each((d, i, nodes) => 
                        self._drawBox(d3.select(nodes[i]), d.obj.path, dims, d.obj.draw.varBoxDims))
                },
                update => {
                    update.transition(sharedTransition)
                        .attr('transform', d =>
                            `translate(${dims.width * d.startI}, ${dims.height * d.startI})`)

                    update.each((d, i, nodes) =>
                        self._drawBox(d3.select(nodes[i]), d.obj.path, dims, d.obj.draw.varBoxDims))
                },
                exit => {
                    exit.transition(sharedTransition)
                        .attr('transform', d => {
                            const transX = dims.width * (d.startI - exitIndex),
                              transY = dims.height * (d.startI - exitIndex);
                            return `translate(${transX}, ${transY})`;
                        })
                        .each((d, i, nodes) => {
                            const exitBoxDims = new Dimensions({ 'count': 1 }, null, d.obj.draw.varBoxDims);
                            self._drawBox(d3.select(nodes[i]), d.obj.path, dims, exitBoxDims);
                            d.obj.draw.varBoxDims.preserve().count = 0;
                        })
                        .remove();                 
                }
            )
    }

    _preDraw(cellDims) {
        this._drawBgSquares(cellDims);
    }
}
