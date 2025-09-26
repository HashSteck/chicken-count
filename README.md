# Chicken Detection Script

A Node.js script and web demo that uses TensorFlow.js and the COCO-SSD model to detect chickens (birds) in images.

## Features

- 🐔 Detects chickens/birds in images using machine learning
- 📊 Shows confidence scores and bounding box coordinates
- 📁 Supports single images or batch processing of directories (Node.js)
- 🖼️ Works with JPEG, PNG, BMP, and GIF formats
- 🚀 Uses TensorFlow.js for fast, client-side inference
- 🌐 Includes both command-line tool and web interface

## Installation

1. Clone or download this project
2. Install dependencies:
   ```bash
   npm install
   ```

## Usage

### Node.js Command Line Tool

#### Using COCO-SSD Model (General Object Detection)

##### Single Image Detection
```bash
node chicken-count.js path/to/your/image.jpg
```

##### Batch Processing (Directory)
```bash
node chicken-count.js path/to/image/directory
```

#### Using Custom Teachable Machine Model

##### Single Image with Custom Model
```bash
# Local model files
node chicken-count-teachable.js ./image.jpg ./model/model.json

# Remote Teachable Machine model
node chicken-count-teachable.js ./image.jpg https://teachablemachine.withgoogle.com/models/YOUR_ID/model.json
```

##### Batch Processing with Custom Model
```bash
node chicken-count-teachable.js ./image-directory/ ./model/model.json
```

### Web Demo

1. Open `web-demo.html` in your web browser
2. Drag and drop an image or click to select one
3. Click "Detect Chickens" to analyze the image
4. View results with bounding boxes drawn on detected chickens

## Example Output (Node.js)

### COCO-SSD Model Output
```
Loading COCO-SSD model...
This may take a moment on first run as the model is downloaded...
Model loaded successfully!
Analyzing image: chicken-photo.jpg

=== DETECTION RESULTS ===
Image: chicken-photo.jpg
Image size: 800x600
Total objects detected: 3
Birds/Chickens detected: 2

Bird/Chicken detections:
  1. Confidence: 87.3%
     Location: x=120, y=200
     Size: 180x240
  2. Confidence: 76.5%
     Location: x=450, y=180
     Size: 150x200

All detected objects:
  1. bird (87.3%)
  2. bird (76.5%)
  3. person (65.2%)
========================

🐔 Found 2 chicken(s) in the image!
```

### Teachable Machine Model Output
```
Loading custom Teachable Machine model...
Model URL: https://teachablemachine.withgoogle.com/models/YOUR_ID/model.json
Custom model loaded successfully!
Model input shape: [,224,224,3]
Model output shape: [,2]
Analyzing image: chicken-photo.jpg

=== TEACHABLE MACHINE RESULTS ===
Image: chicken-photo.jpg
Original image size: 800x600

All predictions:
  No Chicken: 15.2%
  Chicken: 84.8%

Best prediction: Chicken (84.8%)
🐔 CHICKEN DETECTED! Confidence: 84.8%
================================

🎉 Teachable Machine detected a chicken with 84.8% confidence!
```

## How It Works

The script offers two different approaches:

### COCO-SSD Model (`chicken-count.js`)
- **TensorFlow.js**: Machine learning framework for JavaScript
- **COCO-SSD Model**: Pre-trained object detection model that can identify 80 different objects
- **Jimp**: Image processing library for Node.js (command-line version)
- **Browser APIs**: Canvas and Image APIs for the web demo

The COCO-SSD model classifies chickens under the "bird" category, so the script filters for bird detections with confidence scores above 50%.

### Teachable Machine Model (`chicken-count-teachable.js`)
- **Custom Training**: Train your own model specifically for chickens at https://teachablemachine.withgoogle.com/
- **Higher Accuracy**: Specialized for your specific use case
- **Flexible Classes**: Can distinguish between different chicken breeds or other custom categories
- **Smaller Size**: Usually 1-5MB vs 10MB for COCO-SSD

## Creating Your Own Teachable Machine Model

1. **Go to Teachable Machine**: Visit https://teachablemachine.withgoogle.com/
2. **Select Image Project**: Choose "Image Project"
3. **Add Classes**: Create classes like "Chicken", "No Chicken", or specific breeds
4. **Upload Images**: Add 10+ images per class for better accuracy
5. **Train**: Click "Train Model" and wait for completion
6. **Export**: Choose "TensorFlow.js" and copy the model URL
7. **Use**: Run `node chicken-count-teachable.js image.jpg YOUR_MODEL_URL`

## Files

- `chicken-count.js` - Main Node.js script for command-line usage (uses COCO-SSD model)
- `chicken-count-teachable.js` - Version for custom Teachable Machine models
- `web-demo.html` - Interactive web interface for browser-based detection
- `test-installation.js` - Test script to verify installation
- `package.json` - Node.js dependencies and scripts
- `README.md` - This documentation file

## Testing Installation

To verify everything is working correctly:

```bash
node test-installation.js
```

This will load the model and confirm all dependencies are properly installed.

## Limitations

### COCO-SSD Model
- The model detects "birds" in general, not specifically chickens
- Accuracy depends on image quality, lighting conditions, and chicken visibility
- Small or partially obscured chickens may not be detected
- False positives may occur with other bird-like objects or bird-shaped decorations

### Teachable Machine Model
- Accuracy depends on the quality and quantity of training images
- Requires at least 10+ images per class for good results
- May not generalize well to very different environments than training data
- Custom models need to be retrained if requirements change

### General Limitations
- The first run may take longer as the model needs to be downloaded (~10MB for COCO-SSD, 1-5MB for Teachable Machine)
- Internet connection required for initial model download

## Requirements

- Node.js 14+ (for command-line version)
- Modern web browser with JavaScript enabled (for web demo)
- Internet connection for initial model download

## Dependencies

- `@tensorflow/tfjs`: TensorFlow.js core library
- `@tensorflow-models/coco-ssd`: Pre-trained COCO-SSD object detection model
- `jimp`: Image processing for Node.js
- `jsdom`: DOM environment simulation for Node.js

## Troubleshooting

### Model Loading Issues
- Ensure you have a stable internet connection
- The model download is ~10MB and may take time on slower connections
- Clear your browser cache if using the web demo

### Image Processing Issues
- Verify image format is supported (JPEG, PNG, BMP, GIF)
- Check that image file exists and is readable
- Large images may take longer to process

### Performance Tips
- For better performance, use smaller images (under 1MB)
- The web demo runs entirely in your browser - no data is sent to servers
- First detection may be slower due to model initialization

## License

MIT License - Feel free to use and modify as needed!

## Contributing

Feel free to submit issues and enhancement requests!