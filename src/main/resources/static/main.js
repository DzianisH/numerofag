var main = {
    scale: null,
    penSize: null,
    canvSize: null,
    canvas: null,
    ctx: null,

    init: function (sizeX, sizeY, scale, penSize) {
        main.canvas = document.getElementById("myCanvas");
        main.ctx = main.canvas.getContext("2d");
        main.canvSize = {x: sizeX * scale, y: sizeY * scale};
        main.scale = scale;
        if(!penSize) penSize = scale;
        main.penSize = penSize;
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
        main.reshapeAndScale(data);

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
    },

    printFeatures: function (features) {
        var str = "";
        for(var i = 0; i < features.length; ++i){
            for(var j = 0; j < features[i].length; ++j){
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