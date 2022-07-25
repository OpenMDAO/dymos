// <<hpp_insert gen/utils.js>>
// <<hpp_insert gen/defaults.js>>
// <<hpp_insert js/DmLinkageModelData.js>>
// <<hpp_insert js/DmLinkageDiagram.js>>

var sharedTransition = null;

var enterIndex = 0;
var exitIndex = 0;

// The modelData object is generated and populated by n2_viewer.py
let modelData = DmLinkageModelData.uncompressModel(compressedModel);
delete compressedModel;

var n2MouseFuncs = null;

function dmLinkageMain() {
    const linkageDiag = new DmLinkageDiagram(modelData);
    n2MouseFuncs = linkageDiag.getMouseFuncs();

    linkageDiag.update(false);
}

// wintest();
dmLinkageMain();
