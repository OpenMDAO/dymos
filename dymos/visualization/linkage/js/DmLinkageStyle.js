// <<hpp_insert gen/Style.js>>

/**
 * Manage CSS styles for various elements. Adds handling for solvers and OM-specifc types.
 * @typedef DmLinkageStyle
 */
class DmLinkageStyle extends Style {
    // Define colors for each element type. Selected by Isaias Reyes
    static color = {
        ...Style.color,
        'variableCell': 'black',
        'fixedLinkedVariableCell': 'red',
        'fixedUnlinkedVariableCell': '#ffc000',
        'linkageCell': '#00b051',
        'fixedLinkageCell': '#ffc000',
        'root': '#ccc',
        'group': '#ccc',
        'phase': '#ccc',
        'initial': '#bbb',
        'final': '#999'
    };

    /**
     * Initialize the DmLinkageStyle object.
     * @param {Object} svgStyle A reference to the SVG style section, which will be rewritten.
     * @param {Number} fontSize The font size to apply to text styles.
     */
    constructor(svgStyle, fontSize) {
        super(svgStyle, fontSize);
    }

    /**
     * Associate selectors with various style attributes. Adds support for OM components,
     * subsystems, implicit/explicit outputs, solvers, and Auto-IVC inputs.
     * @param {Number} fontSize The font size to apply to text styles.
     * @returns {Object} An object with selectors as keys, and values that are also objects,
     *     with style attributes as keys and values as their settings.
     */
    _createStyleObj(fontSize) {
        const newCssJson = super._createStyleObj(fontSize);

        const DmLinkageCssJson = {
            '#tree > g.root > rect': {
                'fill': DmLinkageStyle.color.root,
                'fill-opacity': '.8',
            },
            '#tree > g.root > text': {
                'transform': 'rotate(-90)'
            },
            '#tree > g.phase > rect': {
                'cursor': 'pointer',
                'fill': DmLinkageStyle.color.phase,
                'fill-opacity': '.8',
            },
            "#tree > g[class*='initial'] > rect": {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': DmLinkageStyle.color.initial,
            },
            "#tree > g[class*='final'] > rect": {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': DmLinkageStyle.color.final,
            },
            '.arrowHead': {
                'stroke-width': 1.5,
                'stroke': '#ddd',
                'fill': '#222'
            },
            'g.variable_box > line': {
                'stroke': Style.color.variableBox,
                'stroke-width': '2',
                'fill': 'none',
                'stroke': 'black'
            },
            '#backgroundRect': {
                'display': 'none'
            },
            'rect.cond-box-0': {
                'fill': '#eee'
            }            ,
            'rect.cond-box-1': {
                'fill': '#ddd'
            },
            'rect.cond-box-2': {
                'fill': '#ddd'
            },
            'rect.cond-box-3': {
                'fill': '#ccc'
            },
            'text.connected-variable-text': {
                'fill': 'white',
                'stroke': 'black',
                'stroke-width': '1px',
                'text-anchor': 'start',
                'font-weight': 'bold',
                'font-family': 'sans-serif'
            }
        };

        return {...newCssJson, ...DmLinkageCssJson};
    }

    /**
     * Based on the element's type and conditionally other info, determine
     * what CSS style is associated.
     * @param {DmLinkageTreeNode} node The item to check.
     * @return {string} The name of an existing CSS class.
     */
    getNodeClass(node) {
        if (node.draw.minimized) return 'minimized';

        switch (node.type) {
            case 'variable': return node.parent.name; // Either 'initial' or 'final'
            case 'condition': return node.name;
            case 'phase': return 'phase';
            case 'root': return 'root'
            case 'filter': return 'filter';
            default:
                throw `CSS class not found for node type: ${node.type}`
        }
    }
}
