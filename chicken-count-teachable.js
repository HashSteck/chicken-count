// Set up DOM environment for TensorFlow.js
const { JSDOM } = require('jsdom');
const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>');
global.document = dom.window.document;
global.window = dom.window;
global.HTMLCanvasElement = dom.window.HTMLCanvasElement;
global.HTMLImageElement = dom.window.HTMLImageElement;
global.ImageData = dom.window.ImageData;

const fs = require('fs');
const path = require('path');

// Setup TensorFlow.js with proper backend
require('@tensorflow/tfjs-backend-cpu');
const tf = require('@tensorflow/tfjs');
const Jimp = require('jimp');

// Set TensorFlow.js backend to CPU
tf.setBackend('cpu');

class TeachableChickenDetector {
    constructor(modelUrl) {
        this.model = null;
        this.modelUrl = modelUrl || './model/model.json'; // Path to your Teachable Machine model
    }

    async loadModel() {
        console.log('Loading custom Teachable Machine model...');
        console.log(`Model URL: ${this.modelUrl}`);
        
        try {
            // Load the Teachable Machine model
            this.model = await tf.loadLayersModel(this.modelUrl);
            console.log('Custom model loaded successfully!');
            
            // Display model info
            console.log(`Model input shape: [${this.model.inputs[0].shape}]`);
            console.log(`Model output shape: [${this.model.outputs[0].shape}]`);
        } catch (error) {
            console.error('Failed to load custom model:', error.message);
            throw error;
        }
    }

    async detectChickens(imagePath) {
        if (!this.model) {
            throw new Error('Model not loaded. Call loadModel() first.');
        }

        if (!fs.existsSync(imagePath)) {
            throw new Error(`Image file not found: ${imagePath}`);
        }

        console.log(`Analyzing image: ${imagePath}`);
        
        // Load and process the image
        const image = await Jimp.read(imagePath);
        
        // Teachable Machine typically expects 224x224 pixels
        const resized = image.resize(224, 224);
        const { width, height } = resized.bitmap;
        
        // Convert image to tensor (normalized 0-1)
        const imageArray = new Float32Array(width * height * 3);
        let pixelIndex = 0;
        
        resized.scan(0, 0, width, height, (x, y, idx) => {
            imageArray[pixelIndex++] = resized.bitmap.data[idx + 0] / 255; // R
            imageArray[pixelIndex++] = resized.bitmap.data[idx + 1] / 255; // G
            imageArray[pixelIndex++] = resized.bitmap.data[idx + 2] / 255; // B
        });

        // Create tensor (add batch dimension)
        const tensor = tf.tensor4d(imageArray, [1, height, width, 3]);
        
        // Perform prediction
        const predictions = await this.model.predict(tensor);
        const predictionData = await predictions.data();
        
        // Clean up tensors
        tensor.dispose();
        predictions.dispose();

        // Interpret results (adjust for your model)
        const classes = ['No Chicken', 'Chicken']; // Adjust these labels to match your model
        const results = [];
        
        for (let i = 0; i < predictionData.length; i++) {
            results.push({
                class: classes[i] || `Class ${i}`,
                confidence: predictionData[i]
            });
        }
        
        // Find best prediction
        const bestPrediction = results.reduce((prev, current) => 
            (prev.confidence > current.confidence) ? prev : current
        );

        return {
            predictions: results,
            bestPrediction: bestPrediction,
            isChicken: bestPrediction.class === 'Chicken' && bestPrediction.confidence > 0.7,
            confidence: bestPrediction.confidence,
            imageSize: { width: image.bitmap.width, height: image.bitmap.height }
        };
    }

    displayResults(results, imagePath) {
        console.log('\n=== TEACHABLE MACHINE RESULTS ===');
        console.log(`Image: ${imagePath}`);
        console.log(`Original image size: ${results.imageSize.width}x${results.imageSize.height}`);
        
        console.log('\nAll predictions:');
        results.predictions.forEach((pred, index) => {
            console.log(`  ${pred.class}: ${(pred.confidence * 100).toFixed(1)}%`);
        });
        
        console.log(`\nBest prediction: ${results.bestPrediction.class} (${(results.bestPrediction.confidence * 100).toFixed(1)}%)`);
        
        if (results.isChicken) {
            console.log(`🐔 CHICKEN DETECTED! Confidence: ${(results.confidence * 100).toFixed(1)}%`);
        } else {
            console.log(`❌ No chicken detected. Best guess: ${results.bestPrediction.class}`);
        }
        
        console.log('================================\n');
    }

    async processImage(imagePath) {
        try {
            const results = await this.detectChickens(imagePath);
            this.displayResults(results, imagePath);
            return results;
        } catch (error) {
            console.error(`Error processing image: ${error.message}`);
            return null;
        }
    }

    async processMultipleImages(imagePaths) {
        const results = [];
        let totalChickens = 0;
        
        console.log(`Processing ${imagePaths.length} images with Teachable Machine model...\n`);
        
        for (const imagePath of imagePaths) {
            const result = await this.processImage(imagePath);
            if (result) {
                results.push({ imagePath, result });
                if (result.isChicken) totalChickens++;
            }
        }
        
        console.log(`\n=== BATCH SUMMARY ===`);
        console.log(`Total images processed: ${results.length}`);
        console.log(`Total chickens detected: ${totalChickens}`);
        console.log(`=====================\n`);
        
        return results;
    }
}

// Main execution function
async function main() {
    // You can pass the model URL as a command-line parameter
    const modelUrl = process.argv[3] || './model/model.json';
    const detector = new TeachableChickenDetector(modelUrl);
    
    try {
        await detector.loadModel();
        
        const imagePath = process.argv[2];
        
        if (!imagePath) {
            console.log('Usage: node chicken-count-teachable.js <image-path> [model-url]');
            console.log('Example: node chicken-count-teachable.js ./test.jpg');
            console.log('Example: node chicken-count-teachable.js ./test.jpg ./my-model/model.json');
            console.log('Example: node chicken-count-teachable.js ./test.jpg https://teachablemachine.withgoogle.com/models/YOUR_MODEL_ID/model.json');
            return;
        }
        
        let stats;
        try {
            stats = fs.lstatSync(imagePath);
        } catch (error) {
            console.error(`Error accessing path: ${error.message}`);
            return;
        }
        
        if (stats.isDirectory()) {
            const files = fs.readdirSync(imagePath)
                .filter(file => /\.(jpe?g|png|bmp|gif)$/i.test(file))
                .map(file => path.join(imagePath, file));
            
            if (files.length === 0) {
                console.log('No image files found in the directory.');
                return;
            }
            
            await detector.processMultipleImages(files);
        } else {
            const results = await detector.processImage(imagePath);
            
            if (results && results.isChicken) {
                console.log(`🎉 Teachable Machine detected a chicken with ${(results.confidence * 100).toFixed(1)}% confidence!`);
            } else if (results) {
                console.log(`🔍 No chicken detected by Teachable Machine.`);
            }
        }
        
    } catch (error) {
        console.error('Error:', error.message);
    }
}

module.exports = TeachableChickenDetector;

if (require.main === module) {
    main();
}