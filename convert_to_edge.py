"""
Convert PyTorch MobileNetV3-Small to Edge Formats
PyTorch → ONNX → TensorFlow → TFLite (INT8 Quantized)
For NXP i.MX RT / eIQ deployment
"""
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from pathlib import Path
import numpy as np
import json

# Paths
MODEL_PATH = Path('/home/ml/CAS/training_combined/best_model.pth')
OUTPUT_DIR = Path('/home/ml/CAS/edge_deployment')
DATA_DIR = Path('/home/ml/CAS/combined_dataset')
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 128
NUM_CLASSES = 12

# Classes
CLASSES = ['bridge', 'center', 'clean', 'donut', 'edge', 'line_break', 
           'line_collapse', 'other', 'pcb_defect', 'random', 'scratch', 'surface_defect']

class MobileNetV3SmallGrayscale(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.mobilenet = models.mobilenet_v3_small(weights=None)
        # Grayscale input
        self.mobilenet.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        # Classifier
        in_features = self.mobilenet.classifier[3].in_features
        self.mobilenet.classifier[3] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.mobilenet(x)

def load_model():
    """Load trained PyTorch model"""
    print("Loading PyTorch model...")
    model = MobileNetV3SmallGrayscale(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    print(f"✓ Loaded from {MODEL_PATH}")
    return model

def export_onnx(model):
    """Export to ONNX format"""
    print("\n" + "="*50)
    print("Step 1: PyTorch → ONNX")
    print("="*50)
    
    dummy_input = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)
    onnx_path = OUTPUT_DIR / 'model.onnx'
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    # Verify
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"✓ ONNX exported: {onnx_path}")
    print(f"✓ Size: {size_mb:.2f} MB")
    return onnx_path

def convert_to_tflite(onnx_path):
    """Convert ONNX to TensorFlow Lite with INT8 quantization"""
    print("\n" + "="*50)
    print("Step 2: ONNX → TensorFlow → TFLite (INT8)")
    print("="*50)
    
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf
    
    # ONNX → TensorFlow SavedModel
    print("Converting ONNX → TensorFlow...")
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_path = OUTPUT_DIR / 'saved_model'
    tf_rep.export_graph(str(tf_path))
    print(f"✓ TensorFlow SavedModel: {tf_path}")
    
    # Get representative dataset for quantization
    print("Preparing calibration data...")
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    cal_dataset = datasets.ImageFolder(DATA_DIR / 'val', transform=transform)
    cal_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=1, shuffle=True)
    
    def representative_dataset():
        for i, (img, _) in enumerate(cal_loader):
            if i >= 200:  # Use 200 samples for calibration
                break
            yield [img.numpy().astype(np.float32)]
    
    # TensorFlow → TFLite with INT8 quantization
    print("Quantizing to INT8...")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    
    tflite_path = OUTPUT_DIR / 'model_int8.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    size_mb = tflite_path.stat().st_size / (1024 * 1024)
    print(f"✓ TFLite INT8: {tflite_path}")
    print(f"✓ Size: {size_mb:.2f} MB")
    
    # Also create float16 version
    print("\nCreating Float16 version...")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_fp16 = converter.convert()
    
    fp16_path = OUTPUT_DIR / 'model_fp16.tflite'
    with open(fp16_path, 'wb') as f:
        f.write(tflite_fp16)
    
    size_fp16 = fp16_path.stat().st_size / (1024 * 1024)
    print(f"✓ TFLite FP16: {fp16_path}")
    print(f"✓ Size: {size_fp16:.2f} MB")
    
    return tflite_path, fp16_path

def verify_tflite(tflite_path):
    """Verify TFLite model works"""
    print("\n" + "="*50)
    print("Step 3: Verification")
    print("="*50)
    
    import tensorflow as tf
    
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    
    # Test inference
    if input_details[0]['dtype'] == np.uint8:
        test_input = np.random.randint(0, 255, input_details[0]['shape'], dtype=np.uint8)
    else:
        test_input = np.random.rand(*input_details[0]['shape']).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"Test inference: OK (output shape: {output.shape})")
    return True

def generate_deployment_files():
    """Generate deployment metadata"""
    print("\n" + "="*50)
    print("Step 4: Deployment Files")
    print("="*50)
    
    # Model metadata
    metadata = {
        "model_name": "SemiDefectClassifier",
        "version": "1.0",
        "input_size": [1, 1, IMG_SIZE, IMG_SIZE],
        "input_format": "NCHW",
        "input_type": "uint8",
        "output_type": "uint8",
        "num_classes": NUM_CLASSES,
        "classes": CLASSES,
        "preprocessing": {
            "resize": [IMG_SIZE, IMG_SIZE],
            "color_mode": "grayscale",
            "normalize": "0-255 to uint8"
        },
        "target_platform": "NXP i.MX RT / eIQ",
        "accuracy": "96.19%"
    }
    
    with open(OUTPUT_DIR / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # C header for class names
    header = '''// Auto-generated class labels for SemiDefectClassifier
#ifndef DEFECT_CLASSES_H
#define DEFECT_CLASSES_H

#define NUM_CLASSES 12

static const char* CLASS_NAMES[NUM_CLASSES] = {
'''
    for i, cls in enumerate(CLASSES):
        header += f'    "{cls}"{"," if i < len(CLASSES)-1 else ""}\n'
    header += '''};

#endif // DEFECT_CLASSES_H
'''
    
    with open(OUTPUT_DIR / 'defect_classes.h', 'w') as f:
        f.write(header)
    
    print(f"✓ model_metadata.json")
    print(f"✓ defect_classes.h")
    
    # Deployment README
    readme = f'''# Edge Deployment Package

## Model: SemiDefectClassifier v1.0

### Performance
- **Accuracy**: 96.19% (12 classes)
- **Input**: 128×128 grayscale (1 channel)
- **Model Size**: See files below

### Files
| File | Description | Size |
|------|-------------|------|
| `model_int8.tflite` | INT8 quantized (edge MCU) | Best for i.MX RT |
| `model_fp16.tflite` | Float16 (GPU/NPU) | Higher precision |
| `model.onnx` | ONNX format | Cross-platform |
| `model_metadata.json` | Model config | - |
| `defect_classes.h` | C header for labels | - |

### Classes (12)
{', '.join(CLASSES)}

### NXP eIQ Integration
1. Open eIQ Model Tool
2. Import `model_int8.tflite`
3. Generate deployment code
4. Copy to MCU project

### Inference Code (Python)
```python
import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="model_int8.tflite")
interpreter.allocate_tensors()

# Preprocess: resize to 128x128, grayscale, normalize to 0-255
img = preprocess(image)
input_data = np.expand_dims(img, axis=0).astype(np.uint8)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
class_id = np.argmax(output)
```
'''
    
    with open(OUTPUT_DIR / 'README.md', 'w') as f:
        f.write(readme)
    
    print(f"✓ README.md")

def main():
    print("="*60)
    print("EDGE MODEL CONVERSION PIPELINE")
    print("MobileNetV3-Small → ONNX → TFLite INT8")
    print("="*60)
    
    # Step 1: Load model
    model = load_model()
    
    # Step 2: Export ONNX
    onnx_path = export_onnx(model)
    
    # Step 3: Convert to TFLite
    try:
        tflite_int8, tflite_fp16 = convert_to_tflite(onnx_path)
        verify_tflite(tflite_int8)
    except Exception as e:
        print(f"⚠ TFLite conversion error: {e}")
        print("Try: pip install onnx-tf tensorflow")
    
    # Step 4: Generate deployment files
    generate_deployment_files()
    
    print("\n" + "="*60)
    print("✅ CONVERSION COMPLETE!")
    print(f"📁 Output: {OUTPUT_DIR}")
    print("="*60)
    print("\nFiles generated:")
    for f in OUTPUT_DIR.iterdir():
        size = f.stat().st_size / 1024
        print(f"  {f.name}: {size:.1f} KB")

if __name__ == "__main__":
    main()
