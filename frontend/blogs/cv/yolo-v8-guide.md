# YOLOv8 Complete Guide: Real-Time Object Detection

YOLO (You Only Look Once) has been the gold standard for real-time object detection since 2016. YOLOv8 by Ultralytics takes it further — better accuracy, simpler API, and native support for detection, segmentation, pose estimation, and classification in one framework.

---

## Why YOLOv8?

| Feature | YOLOv5 | YOLOv8 |
|---|---|---|
| API | PyTorch-based | Unified CLI + Python |
| Tasks | Detection | Detection + Seg + Pose + Classify |
| Anchor-free | ❌ | ✅ |
| mAP (COCO) | 56.8 | 64.9 |
| Ease of use | Good | Excellent |

---

## Installation

```bash
pip install ultralytics
```

That's it. Ultralytics bundles everything you need.

---

## Inference: Detect Objects in an Image

```python
from ultralytics import YOLO
import cv2

# Load pretrained model (downloads automatically)
model = YOLO('yolov8n.pt')  # nano, small, medium, large, xlarge

# Run inference
results = model('path/to/image.jpg')

# Display results
results[0].show()

# Access detections programmatically
for box in results[0].boxes:
    cls = int(box.cls)
    conf = float(box.conf)
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    print(f"Class: {model.names[cls]}, Conf: {conf:.2f}, BBox: ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
```

---

## Real-Time Detection from Webcam

```python
from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True, verbose=False)
    
    for r in results:
        annotated = r.plot()  # draw boxes
    
    cv2.imshow('YOLOv8 Detection', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Training on Custom Dataset

### Step 1: Prepare Dataset (YOLO Format)

```
dataset/
  images/
    train/   ← training images
    val/     ← validation images
  labels/
    train/   ← .txt annotation files
    val/
```

Each `.txt` file for an image:
```
# format: class_id x_center y_center width height (normalized 0-1)
0 0.512 0.489 0.234 0.312
1 0.721 0.334 0.145 0.198
```

### Step 2: Dataset YAML

```yaml
# dataset.yaml
path: ./dataset
train: images/train
val: images/val

names:
  0: person
  1: car
  2: bicycle
```

### Step 3: Train

```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')  # Start from pretrained weights

results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,          # GPU index
    workers=8,
    project='runs/train',
    name='my_detector',
    patience=30,       # Early stopping
    lr0=0.01,
    augment=True,
)
```

---

## Understanding the Metrics

After training, YOLOv8 reports:

- **mAP50** — Mean Average Precision at IoU=0.5 (lenient)
- **mAP50-95** — Mean Average Precision averaged across IoU thresholds 0.5 to 0.95 (strict)
- **Precision** — Of all detected boxes, how many are real objects?
- **Recall** — Of all real objects, how many did we detect?

```
A good baseline target: mAP50 > 0.85 for most production use cases
```

---

## Model Variants

| Model | Size | Speed | mAP50-95 |
|---|---|---|---|
| YOLOv8n | 3.2MB | 80+ FPS | 37.3 |
| YOLOv8s | 11MB | 50+ FPS | 44.9 |
| YOLOv8m | 26MB | 35+ FPS | 50.2 |
| YOLOv8l | 44MB | 20+ FPS | 52.9 |
| YOLOv8x | 68MB | 12+ FPS | 53.9 |

Choose based on your speed vs accuracy trade-off.

---

## Instance Segmentation

```python
model = YOLO('yolov8n-seg.pt')
results = model('image.jpg')

# Access masks
masks = results[0].masks.data  # shape: (N, H, W)
```

---

## Export for Deployment

```python
# Export to ONNX for cross-platform deployment
model.export(format='onnx', dynamic=True)

# Export to TensorRT for maximum GPU speed
model.export(format='engine', device=0)

# Export to CoreML for iOS/macOS
model.export(format='coreml')
```

---

## Tips for Better Results

1. **Data quality > quantity** — 500 high-quality images often beats 5000 poor ones
2. **Augmentation is powerful** — YOLOv8 applies mosaic, mixup, copy-paste by default
3. **Freeze backbone** — For small datasets, freeze the pretrained backbone layers
4. **Use appropriate model size** — Don't use YOLOv8x on an edge device
5. **Balance classes** — Create oversampling or weighted loss for imbalanced datasets

---

*YOLOv8 has made real-time computer vision accessible to every practitioner. With its unified API and strong pretrained models, you can go from zero to a production detector in hours, not weeks.*
