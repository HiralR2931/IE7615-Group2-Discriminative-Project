# IE 7615 Discriminative Deep Learning Project — Group 2

**Members:** Yi-Fan Liu, Pooja Yogesh Patil, Hiral Hitesh Rana, Wen-Hsin Su

The goal of this milestone is to identify which object appears in a single-object image. Each student in the class photographed one unique object under different lighting conditions, angles and backgrounds, which gives a 73-class image classification problem (OBJ001 to OBJ073). We used the final merged dataset and the final Object IDs published by the instructors.

Each object has its own folder, and the folder name is the class label. Background-only shots (files marked _bg_) contain no object and are kept for the Week 2 multi-object composites, so they were excluded here. This leaves 6,964 object images, with 90 to 106 images per class (mean 95.4). The classes are close to balanced, so no class weighting was needed. All images were resized to 224 × 224 pixels.
    
The data was split 70 / 15 / 15 into training (4,874 images), validation (1,045) and test (1,045) sets. The split is stratified by class with a fixed seed (42), so every object appears in all three sets in the same proportions, and every model was trained and evaluated on the same 	images. The test set has only about 14 to 15 images per class and was used only for final reporting, never for training or model selection.

To reflect the variation in how the photos were taken, the training images were augmented on the fly with random horizontal flips, rotation (±8%), zoom (±15%), translation (±10%), brightness (±20%) and contrast (±20%). Augmentation is active only during training.

## Milestone 1: Single-object image ID

73-class image classification (OBJ001–OBJ073). We used 6,964 object images, split 70 / 15 / 15 (stratified, seed 42) into 4,874 train, 1,045 validation and 1,045 test images. Background-only photos (`*_bg_*`) are excluded.

All five models are built, trained and evaluated in one notebook, [`milestone1/Week1_CNN_Comparison.ipynb`](milestone1/Week1_CNN_Comparison.ipynb). The saved outputs show every training run.

| Model | Type | Val acc | Test acc | Top-5 | Macro F1 | Params (M) |
|---|---|---|---|---|---|---|
| **EfficientNetB0** (selected) | Transfer learning, fine-tuned from `block6a` | 98.2% | **97.4%** | 99.7% | 0.974 | 4.14 |
| ResNet50 | Transfer learning, fine-tuned `conv5` | 97.5% | 97.0% | 99.6% | 0.970 | 23.74 |
| MobileNetV2 | Transfer learning, fine-tuned from `block_13` | 96.2% | 96.2% | 99.3% | 0.962 | 2.35 |
| VGG16 | Transfer learning, fine-tuned `block5` | 95.5% | 95.0% | 98.7% | 0.950 | 14.75 |
| Custom CNN | From scratch (5 × Conv-BN-ReLU-MaxPool) | 77.0% | 78.5% | 92.6% | 0.785 | 1.61 |

The final model was chosen by validation accuracy; the test set was used only for reporting.

### Repository contents

```
milestone1/
├── Week1_CNN_Comparison.ipynb      # data pipeline, all 5 models, training, evaluation, error analysis
└── results/
    ├── model_comparison.csv            # metrics for all 5 models
    ├── model_comparison_cpu_latency.csv
    ├── per_class_report_best.csv       # precision / recall / F1 per class (EfficientNetB0)
    ├── class_names.json, split.csv     # class order and the exact train / val / test split
    ├── figures/                        # accuracy comparison, learning curves, confusion matrix, errors
    └── <model>/                        # metrics.json, history.json (per-epoch curves), summary.txt (architecture)
```

### How to run

1. Put the class dataset as `database.zip` in your Google Drive (one folder per object: `database/images_OBJxxx/`). The dataset is not included here because it is the class's shared photo collection.
2. Open the notebook in Google Colab and select **Runtime → Change runtime type → T4 GPU**.
3. In the first code cell, set `DRIVE_DIR` to the Drive folder that contains `database.zip`.
4. **Runtime → Run all.** Training all five models takes about 75 minutes on a T4. Models and results are written to `DRIVE_DIR/week1_results/`.

Built with TensorFlow 2.20 / Keras 3 on Google Colab.

**Notes**
- Trained model files (`.keras`, 19–206 MB each) are not in the repository because of GitHub's 100 MB file limit. Running the notebook recreates them.
- The latency column in `model_comparison.csv` was measured with single-image calls on the GPU, which mostly reflects layer count rather than compute. `model_comparison_cpu_latency.csv` adds a single-image CPU latency measured on a laptop (Intel i5), which is a fairer speed comparison.
