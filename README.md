# Invasive Ductal Carcinoma Classifier

Image classifier for histopathology patches (50x50px) to detect invasive ductal
carcinoma (IDC) in breast tissue, using transfer learning with ResNet50. Includes
the training notebook and a web API (FastAPI) to serve the model.

## Dataset

[Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)
from Kaggle: 277,524 patches (50x50px) extracted from samples of 279 patients, labeled
as IDC positive (class 1) or negative (class 0).

Due to compute constraints (training on Google Colab's free tier), the final model
was trained on a reproducible sample of 120 patients (123,674 images).

## Methodology

- Architecture: ResNet50 pretrained on ImageNet, with fine-tuning of the `conv5_x`
  block and a binary classification head (GlobalAveragePooling2D > Dense > Dropout > Dense > sigmoid).
- Data split: train/val/test split at the patient level (not image level) using
  `GroupShuffleSplit`, to prevent data leakage between sets.
- Class imbalance: handled with balanced `class_weight`.
- Augmentation: rotation, flips, zoom and shifts on the training set.
- Threshold selection: instead of the default 0.5, the threshold was chosen to maximize precision subject to a recall floor of ≥ 0.85 (clinical priority: minimize false negatives).

## Results (test set)

Metric  Value 

Accuracy - 0.86 
Recall (carcinoma class 1) - 0.86 
Precision (carcinoma class) - 0.63 
F1-score (carcinoma class) - 0.72 

These results were confirmed across two independent training runs (same data and
hyperparameters, different weight initialization)

##limitations

- Sample of 120 out of 279 available patients, due to training time constraints on  Colab's free tier.
- 50x50px patches don't include spatial context from surrounding tissue, a known limitation of this type of dataset for histopathology classification.
- Published benchmarks using more modern  architectures (e.g. EfficientNetV2) and the
  full dataset report higher metrics (~0.91-0.97); the gap is likely explained by
  architecture choice, sample size, and available compute resources.


## Running it locally

1. Clone the repository and enter `carcinoma-web/`.
2. Create a virtual environment and install dependencies:
```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   pip install -r requirements.txt
```
3. Download the trained model (~220MB, not included in the repo due to size) from:
   **[PENDING]**, and place it at `carcinoma-web/models/modelo_final.keras`.
4. Run the server:
```bash
   uvicorn main:app --reload
```
5. Open `http://127.0.0.1:8000/static/index.html` in your browser.

