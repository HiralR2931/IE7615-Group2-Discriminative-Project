# IE 7615 Discriminative Deep Learning Project — Group 2

CNN-based automated class attendance system: identify a single-object image by its object ID (Milestone 1), then detect and locate multiple objects in one image (Milestone 2).

**Members:** Yi-Fan Liu, Pooja Yogesh Patil, Hiral Hitesh Rana, Wen-Hsin Su

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
