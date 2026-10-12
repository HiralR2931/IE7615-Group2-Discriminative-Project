# IE 7615 Discriminative Deep Learning Project — Group 2

**Members:** Yi-Fan Liu, Pooja Yogesh Patil, Hiral Hitesh Rana, Wen-Hsin Su

Each student in the class photographed one unique object under different lighting conditions, angles and backgrounds. This gives 73 objects (OBJ001 to OBJ073). We used the final merged dataset and the final Object IDs published by the instructors.

| Milestone | Task | Model | Main result |
|---|---|---|---|
| [Milestone 1](#milestone-1-single-object-image-id) | Identify the object in a single-object image | EfficientNetB0 | 97.4% test accuracy |
| [Milestone 2](#milestone-2-multi-object-detection) | Identify **and locate** every object in a multi-object image | YOLOv8s | 0.909 test mAP@0.5 |

The dataset itself is not included in this repository, because it is the class's shared photo collection. Only code and small result files are stored here.

---

## Milestone 1: Single-object image ID

73-class image classification (OBJ001–OBJ073). Each object has its own folder, and the folder name is the class label. Background-only shots (`*_bg_*`) contain no object and were excluded. This leaves 6,964 object images, with 90 to 106 images per class (mean 95.4). The classes are close to balanced, so no class weighting was needed. All images were resized to 224 × 224 pixels.

The data was split 70 / 15 / 15 into training (4,874 images), validation (1,045) and test (1,045) sets, stratified by class with a fixed seed (42). The test set was used only for final reporting, never for training or model selection.

Training images were augmented on the fly with random horizontal flips, rotation (±8%), zoom (±15%), translation (±10%), brightness (±20%) and contrast (±20%).

All five models are built, trained and evaluated in one notebook, [`milestone1/Week1_CNN_Comparison.ipynb`](milestone1/Week1_CNN_Comparison.ipynb).

| Model | Type | Val acc | Test acc | Top-5 | Macro F1 | Params (M) |
|---|---|---|---|---|---|---|
| **EfficientNetB0** (selected) | Transfer learning, fine-tuned from `block6a` | 98.2% | **97.4%** | 99.7% | 0.974 | 4.14 |
| ResNet50 | Transfer learning, fine-tuned `conv5` | 97.5% | 97.0% | 99.6% | 0.970 | 23.74 |
| MobileNetV2 | Transfer learning, fine-tuned from `block_13` | 96.2% | 96.2% | 99.3% | 0.962 | 2.35 |
| VGG16 | Transfer learning, fine-tuned `block5` | 95.5% | 95.0% | 98.7% | 0.950 | 14.75 |
| Custom CNN | From scratch (5 × Conv-BN-ReLU-MaxPool) | 77.0% | 78.5% | 92.6% | 0.785 | 1.61 |

The final model was chosen by validation accuracy.

---

## Milestone 2: Multi-object detection

Given one image that contains many objects, the model returns the **Object ID and bounding box of every object**. Everything runs in one notebook, [`milestone2/Week2_MultiObject_v2.ipynb`](milestone2/Week2_MultiObject_v2.ipynb).

### Pipeline

1. **Clean and split the single photos.** We used all 73 Object IDs. `OBJ054_017` is excluded (not a photo of the object), and 16 exact duplicates (same decoded pixels) are removed *before* splitting. Every photo is rotated by its EXIF flag and resized to 224 × 224, which fixes `OBJ003_072` and `OBJ023_001`. This leaves 6,947 unique photos, split 70 / 15 / 15 (4,862 / 1,042 / 1,043), stratified, seed 42. A check confirms no photo or copy of it is in more than one split.
2. **Create one bounding box per photo.** The dataset has no boxes, so two open-vocabulary detectors, **YOLO-World v2 (x)** and **Grounding DINO (tiny)**, are prompted with each object's name from the class sheet. When their boxes agree (IoU ≥ 0.5), the final box is the average of the two (76.0% of photos, mean IoU 0.944). Otherwise a single confident box is used. 349 photos (5.0%, from 42 IDs) were left out because neither model found the object reliably; no ID exceeded the 20% limit. Labels were checked with a random sample, a disagreement sample and all photos of OBJ036.
3. **Build grid images.** 224 × 224 photos are placed side by side, using photos from one split only:

   | Grid | Objects | Image size | Train | Val | Test |
   |---|---|---|---|---|---|
   | 2 × 2 | 4 | 448 × 448 | 160 | 30 | 30 |
   | 3 × 3 | 9 | 672 × 672 | 160 | 30 | 30 |
   | 4 × 4 | 16 | 896 × 896 | 160 | 30 | 30 |
   | 5 × 4 | 20 | 1120 × 896 | 160 | 30 | 30 |
   | 5 × 5 | 25 | 1120 × 1120 | 160 | 30 | 30 |
   | **Total** | | | **800** | **150** | **150** |

   Training tiles are randomly mirrored (box mirrored too) and get brightness, contrast and colour changes of up to ±25%. Validation and test grids are not augmented.
4. **Train YOLOv8s** (COCO-pretrained, 11.2 M parameters) for 50 epochs at `imgsz=1120`, `batch=8`, on a Colab T4 (about 41 minutes). Checkpoints are saved to Google Drive, and training resumes automatically after a disconnect.

### Results (150 test grids, 2,220 objects)

| Grid | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| 2 × 2 | 0.842 | 0.841 | 0.929 | 0.858 |
| 3 × 3 | 0.885 | 0.848 | 0.926 | 0.880 |
| 4 × 4 | 0.867 | 0.894 | 0.919 | 0.874 |
| 5 × 4 | 0.891 | 0.865 | 0.898 | 0.844 |
| 5 × 5 | 0.901 | 0.893 | 0.916 | 0.855 |
| **All** | **0.902** | **0.896** | **0.909** | **0.850** |

At a fixed confidence of 0.5, 1,920 of 2,220 test objects (86.5%) are found with the correct ID and a box with IoU ≥ 0.5, and 92.2% of reported detections are correct. Inference takes about 23 ms per 1120 px image on a T4.

**Limitations:** the boxes were created automatically, and our checks suggest about one in ten is wrong, in the test labels as well. The hardest photos (very small, dark or cluttered objects) were left out, and test photos are reused across grids. Test grids are also built the same way as the training grids, so performance on real multi-object photos is likely lower than these numbers.

---

## Repository contents

```
milestone1/
├── Week1_CNN_Comparison.ipynb      # data pipeline, all 5 models, training, evaluation, error analysis
└── results/
    ├── model_comparison.csv            # metrics for all 5 models
    ├── model_comparison_cpu_latency.csv
    ├── per_class_report_best.csv       # precision / recall / F1 per class (EfficientNetB0)
    ├── class_names.json, split.csv     # class order and the exact train / val / test split
    ├── figures/                        # accuracy comparison, learning curves, confusion matrix, errors
    └── <model>/                        # metrics.json, history.json, summary.txt
milestone2/
└── Week2_MultiObject_v2.ipynb      # cleaning, auto-labelling, grid generation, YOLOv8s training, evaluation
```

## How to run

1. Put the class dataset as `database.zip` in your Google Drive (one folder per object: `database/images_OBJxxx/`).
2. Open a notebook in Google Colab and select **Runtime → Change runtime type → T4 GPU**.
3. Set `DRIVE_DIR` in the first code cell to the Drive folder that contains `database.zip`.
4. **Runtime → Run all.**
   - Milestone 1: about 75 minutes. Results go to `DRIVE_DIR/week1_results/`.
   - Milestone 2: about 1.5 hours (labelling ~40 min, training ~41 min). Results go to `DRIVE_DIR/week2_results/`. If Colab disconnects, run all again: the labels are reloaded and training continues from the last epoch.

Built with TensorFlow 2.20 / Keras 3 (Milestone 1) and Ultralytics YOLOv8 + Hugging Face Transformers (Milestone 2) on Google Colab.

**Notes**

- Model weights (`.keras`, `.pt`), grid images and labels are not in the repository because of GitHub's file size limits and the course rule to push code only. Running the notebooks recreates them.
- The latency column in `model_comparison.csv` was measured with single-image calls on the GPU, which mostly reflects layer count rather than compute. `model_comparison_cpu_latency.csv` adds a single-image CPU latency measured on a laptop (Intel i5), which is a fairer speed comparison.
