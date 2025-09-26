// Set up DOM environment for TensorFlow.js
const { JSDOM } = require('jsdom');
const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>');
global.document = dom.window.document;
global.window = dom.window;
global.HTMLCanvasElement = dom.window.HTMLCanvasElement;
global.HTMLImageElement = dom.window.HTMLImageElement;
global.ImageData = dom.window.ImageData;

// Mock fetch for model loading
const fs = require('fs');
const path = require('path');

// Setup TensorFlow.js with proper backend
require('@tensorflow/tfjs-backend-cpu');
const tf = require('@tensorflow/tfjs');
const cocoSsd = require('@tensorflow-models/coco-ssd');
const Jimp = require('jimp');

// Set TensorFlow.js backend to CPU
tf.setBackend('cpu');

class ChickenDetector {
    constructor() {
        this.model = null;
    }

    async loadModel() {
        console.log('Loading COCO-SSD model...');
        console.log('This may take a moment on first run as the model is downloaded...');
        
        try {
            this.model = await cocoSsd.load();
            console.log('Model loaded successfully!');
        } catch (error) {
            console.error('Failed to load model:', error.message);
            throw error;
        }
    }

    async detectChickens(imagePath) {
        if (!this.model) {
            throw new Error('Model not loaded. Call loadModel() first.');
        }

        // Check if image file exists
        if (!fs.existsSync(imagePath)) {
            throw new Error(`Image file not found: ${imagePath}`);
        }

        console.log(`Analyzing image: ${imagePath}`);
        
        // Load and process the image using Jimp
        const image = await Jimp.read(imagePath);
        const { width, height } = image.bitmap;
        
        // Convert image to RGB tensor
        const imageArray = new Float32Array(width * height * 3);
        let pixelIndex = 0;
        
        image.scan(0, 0, width, height, (x, y, idx) => {
            // Normalize pixel values to 0-1 range
            imageArray[pixelIndex++] = image.bitmap.data[idx + 0] / 255; // R
            imageArray[pixelIndex++] = image.bitmap.data[idx + 1] / 255; // G
            imageArray[pixelIndex++] = image.bitmap.data[idx + 2] / 255; // B
        });

        // Create tensor from image data
        const tensor = tf.tensor3d(imageArray, [height, width, 3]);
        
        // Run detection
        const predictions = await this.model.detect(tensor);
        
        // Filter for birds (which includes chickens in COCO dataset)
        // COCO-SSD classifies chickens under "bird" category
        const birdPredictions = predictions.filter(prediction => 
            prediction.class === 'bird' && prediction.score > 0.5
        );

        // Clean up tensor
        tensor.dispose();

        return {
            totalDetections: predictions.length,
            birdDetections: birdPredictions.length,
            allPredictions: predictions,
            birdPredictions: birdPredictions,
            imageSize: { width, height }
        };
    }

    displayResults(results, imagePath) {
        console.log('\n=== DETECTION RESULTS ===');
        console.log(`Image: ${imagePath}`);
        console.log(`Image size: ${results.imageSize.width}x${results.imageSize.height}`);
        console.log(`Total objects detected: ${results.totalDetections}`);
        console.log(`Birds/Chickens detected: ${results.birdDetections}`);
        
        if (results.birdPredictions.length > 0) {
            console.log('\nBird/Chicken detections:');
            results.birdPredictions.forEach((prediction, index) => {
                const [x, y, width, height] = prediction.bbox;
                console.log(`  ${index + 1}. Confidence: ${(prediction.score * 100).toFixed(1)}%`);
                console.log(`     Location: x=${Math.round(x)}, y=${Math.round(y)}`);
                console.log(`     Size: ${Math.round(width)}x${Math.round(height)}`);
            });
        } else {
            console.log('\nNo birds/chickens detected in this image.');
        }

        // Show all detections for reference
        if (results.allPredictions.length > 0) {
            console.log('\nAll detected objects:');
            results.allPredictions.forEach((prediction, index) => {
                console.log(`  ${index + 1}. ${prediction.class} (${(prediction.score * 100).toFixed(1)}%)`);
            });
        }
        console.log('========================\n');
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

    // Batch processing method for multiple images
    async processMultipleImages(imagePaths) {
        const results = [];
        let totalChickens = 0;
        
        console.log(`Processing ${imagePaths.length} images...\n`);
        
        for (const imagePath of imagePaths) {
            const result = await this.processImage(imagePath);
            if (result) {
                results.push({ imagePath, result });
                totalChickens += result.birdDetections;
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
    const detector = new ChickenDetector();
    
    try {
        // Load the model
        await detector.loadModel();
        
        // Get image path from command line arguments
        const imagePath = process.argv[2];
        
        if (!imagePath) {
            console.log('Usage: node chicken-count.js <image-path>');
            console.log('Example: node chicken-count.js ./chicken-image.jpg');
            console.log('');
            console.log('Supported image formats: JPEG, PNG, BMP, GIF');
            return;
        }
        
        // Check if it's a directory or single file
        let stats;
        try {
            stats = fs.lstatSync(imagePath);
        } catch (error) {
            console.error(`Error accessing path: ${error.message}`);
            return;
        }
        
        if (stats.isDirectory()) {
            // Process all images in directory
            const files = fs.readdirSync(imagePath)
                .filter(file => /\.(jpe?g|png|bmp|gif)$/i.test(file))
                .map(file => path.join(imagePath, file));
            
            if (files.length === 0) {
                console.log('No image files found in the directory.');
                return;
            }
            
            await detector.processMultipleImages(files);
        } else {
            // Process single image
            const results = await detector.processImage(imagePath);
            
            if (results && results.birdDetections > 0) {
                console.log(`🐔 Found ${results.birdDetections} chicken(s) in the image!`);
            } else if (results) {
                console.log('🔍 No chickens detected in this image.');
            }
        }
        
    } catch (error) {
        console.error('Error:', error.message);
        console.error('Make sure you have a stable internet connection for model download.');
    }
}

// Export for use as module
module.exports = ChickenDetector;

// Run main function if this file is executed directly
if (require.main === module) {
    main();
}