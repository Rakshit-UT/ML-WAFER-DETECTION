# ML-WAFER-DETECTION
## Semiconductor Defect Classification for Edge Deployment

An Edge-AI powered defect classification system for semiconductor wafer/die inspection, built for the **IESA DeepTech Hackathon**.

---

## 🎯 Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **96.19%** |
| **Classes** | 12 |
| **Model** | MobileNetV3-Small |
| **Parameters** | 1.53M |
| **Edge Format** | ONNX |

---

## 📁 Project Structure

```
├── hackathon_submission/       # Final submission package
│   ├── model.onnx              # Edge model (ONNX)
│   ├── REPORT.md               # Technical report
│   ├── PRESENTATION.md         # 12-slide presentation
│   ├── NXP_EIQ_GUIDE.md        # Deployment guide
│   ├── confusion_matrix.png    # Results
│   ├── per_class_accuracy.png
│   └── defect_classes.h        # C header
├── train_combined.py           # Training script
├── prepare_combined_dataset.py # Dataset preparation
└── convert_to_edge.py          # ONNX conversion
```

---

## 🔧 12 Defect Classes

```
bridge, center, clean, donut, edge, line_break, 
line_collapse, other, pcb_defect, random, scratch, surface_defect
```

---

## 📊 Dataset

Combined 5 public datasets:
- WM-811K (811K wafer maps)
- MixedWM38 (38K images)
- Carinthia SEM (4.5K SEM images)
- SD-Saliency (11K surface defects)
- DeepPCB (3K PCB images)

**Final**: 52,358 balanced training samples

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install torch torchvision onnxruntime

# Inference
python -c "
import onnxruntime as ort
import numpy as np
import cv2

session = ort.InferenceSession('hackathon_submission/model.onnx')
img = cv2.imread('test.png', 0)
img = cv2.resize(img, (128, 128))
input_data = (img.reshape(1,1,128,128).astype(np.float32) - 127.5) / 127.5
output = session.run(None, {'input': input_data})[0]
print(f'Predicted class: {np.argmax(output)}')
"
```

---

## 🎯 Target Platform

- **NXP i.MX RT series** via eIQ Toolkit
- TensorFlow Lite Micro compatible
- <20ms inference latency

---

## 📝 License

MIT License

---

*Built for IESA DeepTech Edge-AI Hackathon 2026*
