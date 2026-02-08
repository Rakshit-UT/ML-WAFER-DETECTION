# NXP eIQ Integration Guide
## Deploying Semiconductor Defect Classifier to i.MX RT

---

## Prerequisites
- NXP MCUXpresso IDE
- NXP eIQ Model Tool (included in MCUXpresso)
- i.MX RT1170 or similar EVK board

---

## Step 1: Import ONNX Model

1. Open **NXP eIQ Model Tool**
2. Click **File → Import → ONNX Model**
3. Select `model.onnx` from this package
4. Review model structure:
   - Input: `[1, 1, 128, 128]` (NCHW, grayscale)
   - Output: `[1, 12]` (class probabilities)

---

## Step 2: Quantize to INT8

1. In eIQ Model Tool, select **Quantization**
2. Choose **INT8 (Full Integer)**
3. Load calibration images:
   - Use 100-200 sample images from dataset
   - Or use provided random calibration data
4. Click **Apply Quantization**
5. Verify accuracy drop is <2%

---

## Step 3: Generate Deployment Artifacts

1. Select **Export → TensorFlow Lite Micro**
2. Target: **i.MX RT1170** (or your board)
3. Output files generated:
   ```
   model.cc          # Model weights as C array
   model.h           # Model header
   model_settings.h  # Input/output config
   ```

---

## Step 4: Integrate with MCU Project

### Add Files
Copy generated files to your MCUXpresso project:
```
src/
├── model.cc
├── model.h
├── model_settings.h
└── defect_classes.h    # From this package
```

### Include in main.c
```c
#include "model.h"
#include "defect_classes.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

// Allocate tensor arena
constexpr int kTensorArenaSize = 100 * 1024;  // 100KB
uint8_t tensor_arena[kTensorArenaSize];

// Initialize interpreter
tflite::MicroInterpreter* interpreter = ...;

// Run inference
void classify_defect(uint8_t* image_128x128) {
    // Copy input
    memcpy(interpreter->input(0)->data.uint8, image_128x128, 128*128);
    
    // Invoke
    interpreter->Invoke();
    
    // Get result
    uint8_t* output = interpreter->output(0)->data.uint8;
    int class_id = 0;
    for (int i = 1; i < 12; i++) {
        if (output[i] > output[class_id]) class_id = i;
    }
    
    printf("Detected: %s\n", CLASS_NAMES[class_id]);
}
```

---

## Step 5: Build & Flash

1. In MCUXpresso, build project
2. Flash to EVK board via USB
3. Connect camera or load test images
4. Monitor serial output for predictions

---

## Model Specifications

| Property | Value |
|----------|-------|
| Input Size | 128×128×1 |
| Input Type | uint8 (0-255) |
| Output Size | 12 |
| Output Type | uint8 |
| RAM Required | ~100KB |
| Flash Required | ~2MB |
| Latency | <20ms @ 600MHz |

---

## Class Mapping

```c
// defect_classes.h
#define NUM_CLASSES 12

static const char* CLASS_NAMES[NUM_CLASSES] = {
    "bridge",
    "center",
    "clean",
    "donut",
    "edge",
    "line_break",
    "line_collapse",
    "other",
    "pcb_defect",
    "random",
    "scratch",
    "surface_defect"
};
```

---

## Preprocessing on MCU

```c
// Resize image to 128x128 (use bilinear or nearest neighbor)
void preprocess(uint8_t* src, int src_w, int src_h, uint8_t* dst) {
    float scale_x = (float)src_w / 128.0f;
    float scale_y = (float)src_h / 128.0f;
    
    for (int y = 0; y < 128; y++) {
        for (int x = 0; x < 128; x++) {
            int src_x = (int)(x * scale_x);
            int src_y = (int)(y * scale_y);
            dst[y * 128 + x] = src[src_y * src_w + src_x];
        }
    }
}
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce tensor arena size |
| Slow inference | Enable CMSIS-NN optimizations |
| Wrong predictions | Check input normalization |
| Model load fail | Verify TFLite Micro version |

---

## Resources

- [NXP eIQ Documentation](https://www.nxp.com/design/software/development-software/eiq-ml-development-environment:EIQ)
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
- [i.MX RT1170 User Guide](https://www.nxp.com/products/processors-and-microcontrollers/arm-microcontrollers/i-mx-rt-crossover-mcus/i-mx-rt1170-crossover-mcu-with-arm-cortex-m7-and-cortex-m4-cores:i.MX-RT1170)
