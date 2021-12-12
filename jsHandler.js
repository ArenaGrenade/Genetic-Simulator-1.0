let mouseCoord = { x: 0, y: 0 };
let mapResolution = {x:220, y:220};
let displayResolution = {x:880, y:880};
let ctx = $("#mainCanvas")[0].getContext("2d");

let backend_url = "http://127.0.0.1:5000/"

function displayRender(gameMap) {
    var canvas = document.createElement("canvas");
    canvas.width = mapResolution.x;
    canvas.height = mapResolution.y;
    var ctx_fake = canvas.getContext("2d");
    var idata = ctx_fake.createImageData(mapResolution.x, mapResolution.y);
    idata.data.set(gameMap)
    ctx_fake.putImageData(idata, 0, 0);
    var image = new Image();
    image.src = canvas.toDataURL();
    ctx.drawImage(canvas, 0, 0, displayResolution.x, displayResolution.y);
}

$(document).ready(function() {
    $.post(backend_url + "init-map", mapResolution, function(data, status) {
        // console.log(data)
        $("#mainCanvas").attr("width", displayResolution.x);
        $("#mainCanvas").attr("height", displayResolution.y);

        var renderedMap = data.renderedMap;
        displayRender(renderedMap);
    })
})

$("#mainCanvas").mousedown(function() {
    console.log("Mouse Touched")
})
