const ChickenDetector = require('./chicken-count.js');

async function testInstallation() {
    console.log('🧪 Testing chicken detection installation...\n');
    
    try {
        const detector = new ChickenDetector();
        await detector.loadModel();
        
        console.log('✅ TensorFlow.js and COCO-SSD model loaded successfully!');
        console.log('✅ All dependencies are working correctly.');
        console.log('✅ Ready to detect chickens in images!\n');
        
        console.log('To test with an actual image, run:');
        console.log('  node chicken-count.js your-image.jpg\n');
        
        console.log('Supported image formats: JPEG, PNG, BMP, GIF');
        
    } catch (error) {
        console.error('❌ Test failed:', error.message);
        console.error('Please make sure all dependencies are installed correctly.');
    }
}

// Run the test
testInstallation();