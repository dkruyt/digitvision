$(document).ready(function() {
    var inputGrid = Array(8).fill().map(() => Array(8).fill(1));  // Initialize with 1 for white cells
    var canvas = document.getElementById('inputCanvas');
    var ctx = canvas.getContext('2d');
    var cellSize = 25;  // Adjust cell size for 200x200 canvas with 8x8 grid
    var extendedLineCanvas = document.getElementById('extendedLineCanvas');
    var extendedLineCtx = extendedLineCanvas.getContext('2d');
    
    var trainingLossChart;
    var validationLossChart;
    var trainingAccuracyChart;
    var validationAccuracyChart;

    function drawInputGrid() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw the coordinate numbers
        ctx.fillStyle = 'black';
        ctx.font = '12px Arial';
        for (var i = 0; i < 8; i++) {
            ctx.fillText(i, i * cellSize + cellSize / 2 - 5, 10);  // Top coordinates
            ctx.fillText(i, 5, i * cellSize + cellSize / 2 + 5);  // Left coordinates
        }

        for (var i = 0; i < 8; i++) {
            for (var j = 0; j < 8; j++) {
                var value = inputGrid[i][j];
                var color = 'rgb(' + Math.round(value * 255) + ', ' + Math.round(value * 255) + ', ' + Math.round(value * 255) + ')';
                ctx.fillStyle = color;
                ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
            }
        }
    }
    function drawExtendedLine() {
        extendedLineCtx.clearRect(0, 0, extendedLineCanvas.width, extendedLineCanvas.height);
        for (var i = 0; i < 64; i++) {
            var row = Math.floor(i / 8);
            var col = i % 8;
            var value = inputGrid[row][col];
            var color = 'rgb(' + Math.round(value * 255) + ', ' + Math.round(value * 255) + ', ' + Math.round(value * 255) + ')';
            extendedLineCtx.fillStyle = color;
            extendedLineCtx.fillRect(i * 5, 0, 4, 4);
        }
    }

    function updateActivations(hiddenActivations, outputActivations) {
        var hiddenHtml = '';
        for (var i = 0; i < hiddenActivations[0].length; i++) {
            var activation = hiddenActivations[0][i];
            var color = getColorMap(activation);
            hiddenHtml += '<div class="neuron hidden-neuron" style="background-color: ' + color + ';" data-index="' + i + '">' + activation.toFixed(3) + '</div>';
        }
        $('#hiddenLayer').html(hiddenHtml);
    
        var outputHtml = '';
        for (var i = 0; i < outputActivations[0].length; i++) {
            var activation = outputActivations[0][i];
            var color = getColorMap(activation);
            outputHtml += '<div class="neuron-container" style="text-align: center;">';
            outputHtml += '<div class="neuron output-neuron" style="background-color: ' + color + ';" data-index="' + i + '">' + activation.toFixed(3) + '</div>';
            outputHtml += '<div class="neuron-number">' + (i) + '</div>'; // Number above the neuron
            outputHtml += '</div>';
        }
        $('#outputLayer').html(outputHtml);
    
        addNeuronClickHandlers();
    }

    function getColorMap(value) {
        value = Math.max(0, Math.min(1, value));
        var hue = (1 - value) * 240;
        return 'hsl(' + hue + ', 100%, 50%)';
    }

    function predict() {
        $.ajax({
            url: '/predict',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ inputGrid: inputGrid }),
            success: function(response) {
                var predictedDigit = response.predictedDigit;
                var hiddenActivations = response.hiddenActivations;
                var outputActivations = response.outputActivations;
                $('#result').text('Result: ' + predictedDigit);
                updateActivations(hiddenActivations, outputActivations);
            }
        });
    }

    function clear() {
        inputGrid = Array(8).fill().map(() => Array(8).fill(1));  // Reset to 1 for white cells
        drawInputGrid();
        drawExtendedLine();
        $('#result').text('');
        $('#hiddenLayer').empty();
        $('#outputLayer').empty();
    }

    function prepareData() {
        $.ajax({
            url: '/training_data',
            method: 'GET',
            success: function(response) {
                showTrainingData(response.trainingData);
            }
        });
    }

    function trainModel(epochs) {
        $.ajax({
            url: '/train',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ epochs: epochs }),
            success: function(response) {
                console.log(response.message);
                predict();
            }
        });
    }

    function clearModelData() {
        $.ajax({
            url: '/clear',
            method: 'POST',
            success: function(response) {
                console.log(response.message);
                predict();
            }
        });
    }

    function showTrainingData(trainingData) {
        var trainingHtml = '';
        for (var i = 0; i < trainingData.length; i++) {
            var img = trainingData[i];
            var imgHtml = '<img src="data:image/png;base64,' + img + '" width="28" height="28" data-index="' + i + '">';
            trainingHtml += imgHtml;
        }
        $('#trainingData').html(trainingHtml);
    }

    function loadTrainingImage(index) {
        $.ajax({
            url: '/load_training_image',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ index: index }),
            success: function(response) {
                inputGrid = response.inputGrid;
                drawInputGrid();
                drawExtendedLine();
                predict();
            }
        });
    }

    function trainSingleExample(digit) {
        $.ajax({
            url: '/train_single',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ inputGrid: inputGrid, digit: digit }),
            success: function(response) {
                console.log('Training on digit ' + digit + ' completed');
                predict();
            }
        });
    }

    function addNeuronClickHandlers() {
        $('.hidden-neuron').click(function() {
            var index = $(this).data('index');
            console.log('Hidden neuron ' + index + ' clicked');
            // Implement the desired functionality here
        });

        $('.output-neuron').click(function() {
            var index = $(this).data('index');
            trainSingleExample(index);
        });
    }

    function initializeCharts() {
        var ctx1 = document.getElementById('trainingLossChart').getContext('2d');
        var ctx2 = document.getElementById('validationLossChart').getContext('2d');
        var ctx3 = document.getElementById('trainingAccuracyChart').getContext('2d');
        var ctx4 = document.getElementById('validationAccuracyChart').getContext('2d');
        
        trainingLossChart = new Chart(ctx1, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Training Loss',
                    data: [],
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1,
                    fill: false
                }]
            },
            options: {
                scales: {
                    x: { beginAtZero: true },
                    y: { beginAtZero: true }
                }
            }
        });
        
        validationLossChart = new Chart(ctx2, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Validation Loss',
                    data: [],
                    borderColor: 'rgba(153, 102, 255, 1)',
                    borderWidth: 1,
                    fill: false
                }]
            },
            options: {
                scales: {
                    x: { beginAtZero: true },
                    y: { beginAtZero: true }
                }
            }
        });
        
        trainingAccuracyChart = new Chart(ctx3, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Training Accuracy',
                    data: [],
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1,
                    fill: false
                }]
            },
            options: {
                scales: {
                    x: { beginAtZero: true },
                    y: { beginAtZero: true }
                }
            }
        });
        
        validationAccuracyChart = new Chart(ctx4, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Validation Accuracy',
                    data: [],
                    borderColor: 'rgba(255, 159, 64, 1)',
                    borderWidth: 1,
                    fill: false
                }]
            },
            options: {
                scales: {
                    x: { beginAtZero: true },
                    y: { beginAtZero: true }
                }
            }
        });
    }

    function updateCharts(metrics) {
        trainingLossChart.data.labels = metrics.epoch;
        trainingLossChart.data.datasets[0].data = metrics.training_loss;
        trainingLossChart.update();
        
        validationLossChart.data.labels = metrics.epoch;
        validationLossChart.data.datasets[0].data = metrics.validation_loss;
        validationLossChart.update();
        
        trainingAccuracyChart.data.labels = metrics.epoch;
        trainingAccuracyChart.data.datasets[0].data = metrics.training_accuracy;
        trainingAccuracyChart.update();
        
        validationAccuracyChart.data.labels = metrics.epoch;
        validationAccuracyChart.data.datasets[0].data = metrics.validation_accuracy;
        validationAccuracyChart.update();
    }

    canvas.addEventListener('mousedown', function(e) {
        var rect = canvas.getBoundingClientRect();
        var x = e.clientX - rect.left;
        var y = e.clientY - rect.top;
        var gridX = Math.floor(x / cellSize);
        var gridY = Math.floor(y / cellSize);
        if (gridX >= 0 && gridX < 8 && gridY >= 0 && gridY < 8) {
            inputGrid[gridY][gridX] = 1 - inputGrid[gridY][gridX];
            drawInputGrid();
            drawExtendedLine();
            predict();
        }
    });

    $('#predictButton').click(predict);
    $('#clearButton').click(clear);
    $('#prepareButton').click(prepareData);
    $('#train1Button').click(function() { trainModel(1); });
    $('#train10Button').click(function() { trainModel(10); });
    $('#train100Button').click(function() { trainModel(100); });
    $('#clearModelButton').click(clearModelData);

    $(document).on('click', '#trainingData img', function() {
        var index = $(this).data('index');
        loadTrainingImage(index);
    });

    // Socket.IO client
    var socket = io();

    socket.on('log', function(data) {
        var logWindow = document.getElementById('logWindow');
        logWindow.innerHTML += '<p>' + data.message + '</p>';
        logWindow.scrollTop = logWindow.scrollHeight;
    });

    socket.on('training_metrics', function(metrics) {
        updateCharts(metrics);
    });

    $('#helpButton').click(function() {
        $('#helpModal').modal('show');
    });

    $('#graphsButton').click(function() {
        $('#graphsModal').modal('show');
    });

    drawInputGrid();
    drawExtendedLine();
    initializeCharts();
});
