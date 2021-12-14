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

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms))
}

$(document).ready(async function() {
    $.post(backend_url + "init-map", mapResolution, function(data, status) {
        // console.log(data)
        $("#mainCanvas").attr("width", displayResolution.x);
        $("#mainCanvas").attr("height", displayResolution.y);

        displayRender(data.renderedMap);
    })

    while (true) {
        const response = await fetch(backend_url + "run-timestep");
        const data = await response.json();
        if (data != null && data.renderedMap != null) {
            displayRender(data.renderedMap);
            $('#hrs').text("Current Hour\t" + data.timeStep);
            $('#days').text("Current Day\t" + data.dayNum);
            $('#gen').text("Current Generation\t" + data.generation);
            $('#stats').text(JSON.stringify(data.stats, null, '\n'));
        }
        await delay(2)
    }
})

$("#mainCanvas").mousedown(function() {
    console.log("Mouse Touched")
})
