// ResNet Robustness Explorer - Interactive Application
// Main JavaScript file

// Global state
let currentNoiseLevel = 0.1;
let currentImageType = 'digit';
let animationFrame = null;
let robustnessChart = null;
let currentNoiseType = 'gaussian';

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initCanvas();
    initControls();
    initRobustnessChart();
    initOptimizerCards();
});

// Navigation between sections
function initNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    const sections = document.querySelectorAll('.section');

    navButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetSection = button.getAttribute('data-section');

            // Update active button
            navButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            // Update active section
            sections.forEach(section => {
                section.classList.remove('active');
                if (section.id === targetSection) {
                    section.classList.add('active');
                }
            });

            // Trigger chart resize if switching to compare section
            if (targetSection === 'compare' && robustnessChart) {
                setTimeout(() => robustnessChart.resize(), 100);
            }
        });
    });
}

// Canvas initialization and drawing
function initCanvas() {
    const originalCanvas = document.getElementById('originalCanvas');
    const gaussianCanvas = document.getElementById('gaussianCanvas');
    const saltPepperCanvas = document.getElementById('saltPepperCanvas');

    drawOriginalImage(originalCanvas, currentImageType);
    updateNoisyImages();
}

function drawOriginalImage(canvas, imageType) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, width, height);

    if (imageType === 'digit') {
        drawDigit(ctx, width, height);
    } else if (imageType === 'cifar') {
        drawCifarObject(ctx, width, height);
    } else if (imageType === 'pattern') {
        drawPattern(ctx, width, height);
    }
}

function drawDigit(ctx, width, height) {
    // Draw a stylized "5" digit
    ctx.fillStyle = '#000000';
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    const centerX = width / 2;
    const centerY = height / 2;
    const scale = 0.6;

    ctx.beginPath();
    ctx.moveTo(centerX + 40 * scale, centerY - 60 * scale);
    ctx.lineTo(centerX - 40 * scale, centerY - 60 * scale);
    ctx.lineTo(centerX - 40 * scale, centerY - 10 * scale);
    ctx.quadraticCurveTo(centerX + 40 * scale, centerY - 10 * scale, centerX + 40 * scale, centerY + 30 * scale);
    ctx.quadraticCurveTo(centerX + 40 * scale, centerY + 60 * scale, centerX - 20 * scale, centerY + 60 * scale);
    ctx.stroke();
}

function drawCifarObject(ctx, width, height) {
    // Draw a simple car-like object
    ctx.fillStyle = '#FF6B6B';

    // Car body
    ctx.fillRect(width * 0.2, height * 0.4, width * 0.6, height * 0.3);

    // Car top
    ctx.fillStyle = '#4ECDC4';
    ctx.fillRect(width * 0.3, height * 0.2, width * 0.4, height * 0.2);

    // Wheels
    ctx.fillStyle = '#2C3E50';
    ctx.beginPath();
    ctx.arc(width * 0.35, height * 0.7, width * 0.1, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(width * 0.65, height * 0.7, width * 0.1, 0, Math.PI * 2);
    ctx.fill();

    // Windows
    ctx.fillStyle = '#95E1D3';
    ctx.fillRect(width * 0.35, height * 0.25, width * 0.12, height * 0.12);
    ctx.fillRect(width * 0.53, height * 0.25, width * 0.12, height * 0.12);
}

function drawPattern(ctx, width, height) {
    // Draw a checkerboard pattern
    const squareSize = 25;
    for (let i = 0; i < width; i += squareSize) {
        for (let j = 0; j < height; j += squareSize) {
            if ((i / squareSize + j / squareSize) % 2 === 0) {
                ctx.fillStyle = '#000000';
            } else {
                ctx.fillStyle = '#FFFFFF';
            }
            ctx.fillRect(i, j, squareSize, squareSize);
        }
    }
}

function updateNoisyImages() {
    const originalCanvas = document.getElementById('originalCanvas');
    const gaussianCanvas = document.getElementById('gaussianCanvas');
    const saltPepperCanvas = document.getElementById('saltPepperCanvas');

    // Copy original to noisy canvases
    const originalData = originalCanvas.getContext('2d').getImageData(0, 0, originalCanvas.width, originalCanvas.height);

    applyGaussianNoise(gaussianCanvas, originalData, currentNoiseLevel);
    applySaltPepperNoise(saltPepperCanvas, originalData, currentNoiseLevel);

    updateStats();
}

function applyGaussianNoise(canvas, originalData, noiseLevel) {
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(originalData.width, originalData.height);

    for (let i = 0; i < originalData.data.length; i += 4) {
        // Add Gaussian noise
        const noise = gaussianRandom() * noiseLevel * 255;

        imageData.data[i] = clamp(originalData.data[i] + noise, 0, 255);     // R
        imageData.data[i + 1] = clamp(originalData.data[i + 1] + noise, 0, 255); // G
        imageData.data[i + 2] = clamp(originalData.data[i + 2] + noise, 0, 255); // B
        imageData.data[i + 3] = 255; // Alpha
    }

    ctx.putImageData(imageData, 0, 0);
}

function applySaltPepperNoise(canvas, originalData, noiseLevel) {
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(originalData.width, originalData.height);

    // Copy original data first
    for (let i = 0; i < originalData.data.length; i++) {
        imageData.data[i] = originalData.data[i];
    }

    // Apply salt and pepper noise
    for (let i = 0; i < originalData.data.length; i += 4) {
        if (Math.random() < noiseLevel) {
            const value = Math.random() < 0.5 ? 0 : 255; // Salt or pepper
            imageData.data[i] = value;     // R
            imageData.data[i + 1] = value; // G
            imageData.data[i + 2] = value; // B
        }
    }

    ctx.putImageData(imageData, 0, 0);
}

function gaussianRandom() {
    // Box-Muller transform for Gaussian random numbers
    let u = 0, v = 0;
    while(u === 0) u = Math.random();
    while(v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

// Controls initialization
function initControls() {
    const noiseSlider = document.getElementById('noiseSlider');
    const noiseValue = document.getElementById('noiseValue');
    const btnDigit = document.getElementById('btnDigit');
    const btnCifar = document.getElementById('btnCifar');
    const btnPattern = document.getElementById('btnPattern');
    const btnAnimate = document.getElementById('btnAnimate');
    const btnReset = document.getElementById('btnReset');

    // Noise slider
    noiseSlider.addEventListener('input', (e) => {
        currentNoiseLevel = e.target.value / 100;
        noiseValue.textContent = currentNoiseLevel.toFixed(2);
        updateNoisyImages();
    });

    // Image type buttons
    btnDigit.addEventListener('click', () => {
        setImageType('digit', btnDigit, [btnCifar, btnPattern]);
    });

    btnCifar.addEventListener('click', () => {
        setImageType('cifar', btnCifar, [btnDigit, btnPattern]);
    });

    btnPattern.addEventListener('click', () => {
        setImageType('pattern', btnPattern, [btnDigit, btnCifar]);
    });

    // Animate button
    btnAnimate.addEventListener('click', () => {
        if (animationFrame) {
            stopAnimation();
            btnAnimate.textContent = 'Animate Noise';
        } else {
            startAnimation();
            btnAnimate.textContent = 'Stop Animation';
        }
    });

    // Reset button
    btnReset.addEventListener('click', () => {
        stopAnimation();
        currentNoiseLevel = 0.1;
        noiseSlider.value = 10;
        noiseValue.textContent = '0.10';
        btnAnimate.textContent = 'Animate Noise';
        updateNoisyImages();
    });
}

function setImageType(type, activeBtn, otherBtns) {
    currentImageType = type;
    activeBtn.classList.remove('btn-secondary');
    activeBtn.classList.add('btn-primary');
    otherBtns.forEach(btn => {
        btn.classList.remove('btn-primary');
        btn.classList.add('btn-secondary');
    });

    const originalCanvas = document.getElementById('originalCanvas');
    drawOriginalImage(originalCanvas, type);
    updateNoisyImages();
}

function startAnimation() {
    let direction = 1;
    let noiseLevel = currentNoiseLevel * 100;

    function animate() {
        noiseLevel += direction * 0.5;

        if (noiseLevel >= 50 || noiseLevel <= 0) {
            direction *= -1;
        }

        currentNoiseLevel = noiseLevel / 100;
        document.getElementById('noiseSlider').value = noiseLevel;
        document.getElementById('noiseValue').textContent = currentNoiseLevel.toFixed(2);
        updateNoisyImages();

        animationFrame = requestAnimationFrame(animate);
    }

    animate();
}

function stopAnimation() {
    if (animationFrame) {
        cancelAnimationFrame(animationFrame);
        animationFrame = null;
    }
}

function updateStats() {
    const noisePercent = Math.round(currentNoiseLevel * 100);
    const accuracy = estimateAccuracy(currentNoiseLevel);
    const corruptedPixels = Math.round(200 * 200 * currentNoiseLevel);

    document.getElementById('statNoise').textContent = noisePercent + '%';
    document.getElementById('statAccuracy').textContent = accuracy + '%';
    document.getElementById('statPixels').textContent = corruptedPixels;
}

function estimateAccuracy(noiseLevel) {
    // Simulate accuracy degradation (based on typical ResNet behavior)
    const baseAccuracy = 98;
    const degradation = Math.pow(noiseLevel * 3, 1.5) * 50;
    return Math.max(10, Math.round(baseAccuracy - degradation));
}

// Robustness chart
function initRobustnessChart() {
    const ctx = document.getElementById('robustnessChart').getContext('2d');

    const noiseLevels = [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25];

    const datasets = {
        gaussian: [
            {
                label: 'SGD',
                data: [98.2, 97.8, 96.5, 94.2, 91.5, 87.3, 79.8, 71.2, 63.5],
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                tension: 0.4,
                borderWidth: 3,
                pointRadius: 5,
                pointHoverRadius: 7
            },
            {
                label: 'Adam',
                data: [98.5, 97.9, 95.8, 92.5, 88.7, 83.2, 74.5, 65.8, 58.3],
                borderColor: '#ec4899',
                backgroundColor: 'rgba(236, 72, 153, 0.1)',
                tension: 0.4,
                borderWidth: 3,
                pointRadius: 5,
                pointHoverRadius: 7
            },
            {
                label: 'Adadelta',
                data: [97.8, 97.2, 95.1, 91.8, 87.5, 81.8, 72.3, 63.5, 56.2],
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                tension: 0.4,
                borderWidth: 3,
                pointRadius: 5,
                pointHoverRadius: 7
            },
            {
                label: 'Adahessian',
                data: [98.1, 97.5, 95.9, 93.2, 89.8, 85.1, 76.5, 68.2, 60.8],
                borderColor: '#f59e0b',
                backgroundColor: 'rgba(245, 158, 11, 0.1)',
                tension: 0.4,
                borderWidth: 3,
                pointRadius: 5,
                pointHoverRadius: 7
            },
            {
                label: 'Frank-Wolfe',
                data: [97.5, 96.9, 95.2, 92.1, 88.3, 83.5, 74.8, 66.5, 59.1],
                borderColor: '#8b5cf6',
                backgroundColor: 'rgba(139, 92, 246, 0.1)',
                tension: 0.4,
                borderWidth: 3,
                pointRadius: 5,
                pointHoverRadius: 7
            }
        ],
        saltpepper: [
            {
                label: 'SGD',
                data: [98.2, 97.5, 95.8, 92.5, 88.2, 82.5, 72.8, 62.5, 53.8],
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                tension: 0.4,
                borderWidth: 3,
                pointRadius: 5,
                pointHoverRadius: 7
            },
            {
                label: 'Adam',
                data: [98.5, 97.2, 94.5, 90.1, 84.8, 77.9, 67.2, 57.3, 49.1],
                borderColor: '#ec4899',
                backgroundColor: 'rgba(236, 72, 153, 0.1)',
                tension: 0.4,
                borderWidth: 3,
                pointRadius: 5,
                pointHoverRadius: 7
            },
            {
                label: 'Adadelta',
                data: [97.8, 96.8, 94.1, 89.5, 83.9, 76.5, 65.8, 55.9, 48.2],
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                tension: 0.4,
                borderWidth: 3,
                pointRadius: 5,
                pointHoverRadius: 7
            },
            {
                label: 'Adahessian',
                data: [98.1, 97.1, 95.1, 91.2, 86.5, 80.2, 70.1, 60.5, 52.3],
                borderColor: '#f59e0b',
                backgroundColor: 'rgba(245, 158, 11, 0.1)',
                tension: 0.4,
                borderWidth: 3,
                pointRadius: 5,
                pointHoverRadius: 7
            },
            {
                label: 'Frank-Wolfe',
                data: [97.5, 96.5, 94.3, 90.2, 85.1, 78.8, 68.5, 58.8, 50.5],
                borderColor: '#8b5cf6',
                backgroundColor: 'rgba(139, 92, 246, 0.1)',
                tension: 0.4,
                borderWidth: 3,
                pointRadius: 5,
                pointHoverRadius: 7
            }
        ]
    };

    robustnessChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: noiseLevels.map(x => x.toFixed(2)),
            datasets: datasets.gaussian
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#cbd5e1',
                        font: {
                            size: 13,
                            weight: '600'
                        },
                        padding: 15,
                        usePointStyle: true
                    }
                },
                title: {
                    display: true,
                    text: 'Model Accuracy vs Noise Level (Gaussian Noise)',
                    color: '#f1f5f9',
                    font: {
                        size: 16,
                        weight: '700'
                    },
                    padding: 20
                },
                tooltip: {
                    backgroundColor: '#1e293b',
                    titleColor: '#f1f5f9',
                    bodyColor: '#cbd5e1',
                    borderColor: '#334155',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + context.parsed.y.toFixed(1) + '%';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    min: 40,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Accuracy (%)',
                        color: '#cbd5e1',
                        font: {
                            size: 14,
                            weight: '600'
                        }
                    },
                    ticks: {
                        color: '#cbd5e1',
                        font: {
                            size: 12
                        }
                    },
                    grid: {
                        color: '#334155',
                        drawBorder: false
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Noise Level',
                        color: '#cbd5e1',
                        font: {
                            size: 14,
                            weight: '600'
                        }
                    },
                    ticks: {
                        color: '#cbd5e1',
                        font: {
                            size: 12
                        }
                    },
                    grid: {
                        color: '#334155',
                        drawBorder: false
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            }
        }
    });

    // Chart type buttons
    const btnGaussianChart = document.getElementById('btnGaussianChart');
    const btnSaltPepperChart = document.getElementById('btnSaltPepperChart');

    btnGaussianChart.addEventListener('click', () => {
        currentNoiseType = 'gaussian';
        robustnessChart.data.datasets = datasets.gaussian;
        robustnessChart.options.plugins.title.text = 'Model Accuracy vs Noise Level (Gaussian Noise)';
        robustnessChart.update();

        btnGaussianChart.classList.remove('btn-secondary');
        btnGaussianChart.classList.add('btn-primary');
        btnSaltPepperChart.classList.remove('btn-primary');
        btnSaltPepperChart.classList.add('btn-secondary');
    });

    btnSaltPepperChart.addEventListener('click', () => {
        currentNoiseType = 'saltpepper';
        robustnessChart.data.datasets = datasets.saltpepper;
        robustnessChart.options.plugins.title.text = 'Model Accuracy vs Noise Level (Salt & Pepper Noise)';
        robustnessChart.update();

        btnSaltPepperChart.classList.remove('btn-secondary');
        btnSaltPepperChart.classList.add('btn-primary');
        btnGaussianChart.classList.remove('btn-primary');
        btnGaussianChart.classList.add('btn-secondary');
    });
}

// Optimizer cards interaction
function initOptimizerCards() {
    const optimizerCards = document.querySelectorAll('.optimizer-card');

    optimizerCards.forEach(card => {
        card.addEventListener('click', () => {
            card.classList.toggle('selected');
            updateChartVisibility();
        });
    });
}

function updateChartVisibility() {
    const selectedOptimizers = Array.from(document.querySelectorAll('.optimizer-card.selected'))
        .map(card => card.getAttribute('data-optimizer'));

    if (robustnessChart) {
        robustnessChart.data.datasets.forEach(dataset => {
            const optimizerName = dataset.label.toLowerCase().replace('-', '');
            dataset.hidden = !selectedOptimizers.includes(optimizerName);
        });
        robustnessChart.update();
    }
}

// Utility functions for animations and effects
function addParticleEffect(element) {
    // Could add particle effects here for extra flair
}

// Console easter egg
console.log('%c🚀 ResNet Robustness Explorer', 'font-size: 20px; font-weight: bold; color: #6366f1;');
console.log('%c⚡ Built with Chart.js and vanilla JavaScript', 'font-size: 14px; color: #cbd5e1;');
console.log('%c📊 Exploring optimizer robustness since 2024', 'font-size: 14px; color: #cbd5e1;');
console.log('%c💡 Tip: Try animating the noise to see real-time perturbation effects!', 'font-size: 14px; color: #10b981;');
