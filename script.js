document.addEventListener('DOMContentLoaded', function () {
    const backgroundColorInput = document.getElementById('backgroundColor');
    document.body.style.backgroundColor = backgroundColorInput.value;
});

const video = document.getElementById('video');
const videoContainer = document.getElementById('videoContainer');
const predictedLetterElement = document.getElementById('predictedLetter');

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
        return video.play();
    })
    .catch(error => {
        console.error('Error accessing webcam:', error);
    });

let isVowelMode = true; // Initial mode is vowel

// Function to toggle between vowel and consonant modes
function toggleMode() {
    isVowelMode = !isVowelMode;
    const changeModeButton = document.getElementById('changeMode');
    changeModeButton.textContent = isVowelMode ? 'Switch to Consonant' : 'Switch to Vowel';
}

// Load your machine learning models (replace with your actual TensorFlow.js code)
async function loadModel() {
    // Load your vowel and consonant models
    // Example:
    // vowelModel = await tf.loadLayersModel('path/to/vowel/model.json');
    // consonantModel = await tf.loadLayersModel('path/to/consonant/model.json');
}

// Function to predict the letter based on the current mode
async function predictLetter() {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = video.width;
    canvas.height = video.height;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Get the image data from the canvas
    const imageData = context.getImageData(0, 0, canvas.width, canvas.height);

    // Preprocess the image if needed (resize, normalize, etc.)

    // Load your machine learning models
    await loadModel();

    // Run your machine learning models based on the current mode
    let prediction = 'NON'; // Default value if the model output is not available
    if (isVowelMode) {
        // Example: Replace with your actual prediction logic for vowel
        // const output = await vowelModel.predict(preprocessedImageData);
        // prediction = processOutput(output);
    } else {
        // Example: Replace with your actual prediction logic for consonant
        // const output = await consonantModel.predict(preprocessedImageData);
        // prediction = processOutput(output);
    }

    // Display the result
    predictedLetterElement.textContent = prediction;

    // Continue capturing and predicting
    requestAnimationFrame(predictLetter);
}

// Start capturing and predicting once the webcam stream is loaded
video.addEventListener('loadeddata', () => {
    predictLetter();
});

// Event listener for the mode change button
const changeModeButton = document.getElementById('changeMode');
changeModeButton.addEventListener('click', toggleMode);

// Event listener for background color selection
const backgroundColorInput = document.getElementById('backgroundColor');
backgroundColorInput.addEventListener('input', () => {
    document.body.style.backgroundColor = backgroundColorInput.value;
});