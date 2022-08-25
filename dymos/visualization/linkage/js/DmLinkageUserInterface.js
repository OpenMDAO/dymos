// <<hpp_insert gen/UserInterface.js>>
// <<hpp_insert js/DmLinkageLegend.js>>

/**
 * Handle input events for the matrix and toolbar.
 * @typedef DmLinkageUserInterface
 */

class DmLinkageUserInterface extends UserInterface {
    /**
     * Initialize properties, set up the collapse-depth menu, and set up other
     * elements of the toolbar.
     * @param {OmDiagram} diag A reference to the main diagram.
     */
    constructor(diag) {
        super(diag);
    }

    /**
     * Separate these calls from the constructor so that subclasses can
     * set values before execution.
     */
    _init() {
        this.legend = new DmLinkageLegend(this.diag.modelData);
        this.toolbar = new Toolbar(this);
    }
}
