var main = {
    scale: null,
    penSize: null,
    canvSize: null,
    ctx: null,
    answer: null,
    previewCtx: null,
    probabilities: null,

    init: function (sizeX, sizeY, scale, penSize) {
        main.answer = document.getElementById("answer");
        main.previewCtx = document.getElementById("previewCanvas").getContext("2d");
        main.ctx = document.getElementById("inputCanvas").getContext("2d");
        main.canvSize = {x: sizeX * scale, y: sizeY * scale};
        main.scale = scale;
        if (!penSize) penSize = scale;
        main.penSize = penSize;

        main.probabilities = new Array(10);
        for (var i = 0; i < main.probabilities.length; ++i) {
            main.probabilities[i] = document.getElementById("probability-for-" + i);
        }
    },

    onMouseMove: function (event) {
        if (event.buttons === 1) {
            main.onMouseDrag(main.getPosition(event));
        }
    },

    getPosition: function (event) {
        return {x: event.layerX, y: event.layerY};
    },

    onMouseDrag: function (position) {
        main.ctx.fillRect(position.x, position.y, main.penSize, main.penSize);
    },

    clear: function () {
        main.ctx.clearRect(0, 0, main.canvSize.x, main.canvSize.y);//TODO:
    },

    eval: function () {
        var data = main.extractColorsArray();
        data = main.reshapeAndScale(data);
        main.updatePreview(data);
        this.sendRequest(data);
    },

    sendRequest: function (data) {
        var xhr = new XMLHttpRequest();
        xhr.open("POST", "", true);
        xhr.setRequestHeader("Content-Type", "application/json");
        xhr.onload = function (e) {
            if (xhr.readyState === 4) {
                if (xhr.status === 200) {
                    var response = JSON.parse(xhr.responseText);
                    var predictionEstimate = response.estimates[response.predictionClass];
                    var estimate = Math.round(predictionEstimate * 100);
                    main.answer.innerHTML = "You Drew " + response.predictionClass + " (" + estimate + "% sure)";

                    for (var i = 0; i < response.estimates.length; ++i) {
                        estimate = response.estimates[i];
                        main.probabilities[i].innerHTML = estimate.toFixed(3);

                        var classes = "";
                        if(i === response.predictionClass) classes = "predicted-value ";
                        if(estimate / predictionEstimate >= 0.5) classes += "prediction-noise";
                        main.probabilities[i].parentNode.parentNode.setAttribute("class", classes);
                    }

                } else {
                    console.error(xhr);
                    main.answer.innerHTML = "Unexpected Error Occurred";
                }
            }
        };
        xhr.onerror = function (e) {
            console.error(xhr);
            main.answer.innerHTML = "Unexpected Error Occurred";
        };

        main.answer.innerHTML = "Executing...";
        xhr.send(JSON.stringify(data));
    },

    updatePreview: function (features) {
        var imgData = main.previewCtx.createImageData(main.canvSize.x, main.canvSize.y);
        for (var i = 0; i < features.length; ++i) {
            for (var j = 0; j < features[i].length; ++j) {
                for (var k = 0; k < main.scale; ++k) {
                    for (var w = 0; w < main.scale; ++w) {
                        var x = (i * main.scale + k) * main.scale;
                        var y = j * main.scale + w;
                        var index = (x * features[i].length + y) * 4 + 3;
                        imgData.data[index] = Math.floor(features[i][j] * 255);
                    }
                }
            }
        }

        main.previewCtx.putImageData(imgData, 0, 0);

    },

    extractColorsArray: function () {
        var data = main.ctx.getImageData(0, 0, main.canvSize.x, main.canvSize.y).data;
        data = data.filter(function (_, index) {
            return index % 4 === 3; // i don't care about rgb
        });
        return data;
    },

    reshapeAndScale: function (data) {
        var result = main.createResultArray();
        var sqScale = main.scale * main.scale;
        var crlCoef = sqScale * 255; // 255 is max color brightness

        for (var i = 0; i < main.canvSize.x; ++i) {
            for (var j = 0; j < main.canvSize.y; ++j) {
                var index = i * main.canvSize.y + j;
                var resI = Math.floor(i / main.scale);
                var resJ = Math.floor(j / main.scale);

                result[resI][resJ] += data[index] / crlCoef;
            }
        }
        main.printFeatures(result);
        return result;
    },

    printFeatures: function (features) {
        var str = "";
        for (var i = 0; i < features.length; ++i) {
            for (var j = 0; j < features[i].length; ++j) {
                str += features[i][j].toFixed(2) + " ";
            }
            str += "\n";
        }
        str += "\n";
        console.log(str);
    },

    createResultArray: function () {
        var result = new Array(main.canvSize.x / main.scale);
        var sizeY = main.canvSize.y / main.scale;
        for (var i = 0; i < result.length; ++i) {
            result[i] = new Array(sizeY);
            for (var j = 0; j < sizeY; ++j) {
                result[i][j] = 0.0;
            }
        }
        return result;
    }
};