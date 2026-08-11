<div style="background-color: #fff0f5; padding: 25px; border-radius: 12px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #2d2d2d; border: 1px solid #f8bbd0;">

# Breast Cancer Detection & Classification Model

Live Demo: [https://breast-cancer-detection-model-srinivasan.vercel.app/](https://breast-cancer-detection-model-srinivasan.vercel.app/)

---

## Research Background & Clinical Domain

Breast cancer diagnosis relies heavily on **Fine Needle Aspiration (FNA)** biopsy, a minimally invasive procedure where a thin needle extracts cell samples directly from suspicious breast tissue masses. The extracted cell nuclei are digitized via high-resolution microscopic imaging and analyzed morphometrically.

Quantitative features derived from digital images of cell nuclei capture subtle structural and texture changes:
- **Benign Lesions** (non-cancerous, e.g., fibroadenomas or cysts) consist of uniform, smooth, round or oval nuclei with consistent chromatin distribution and regular nuclear envelopes.
- **Malignant Carcinomas** exhibit marked pleomorphism — enlarged nuclei, irregular and notched nuclear contours, variable surface texture, and increased nuclear density due to aggressive cellular proliferation.

This project implements machine learning classification models trained on digitized FNA nuclear measurements to accurately differentiate benign tissue from malignant carcinomas.

Based upon the [UCI Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).

---

## What the Model Finds Out

Through feature importance analysis and cross-validation across 569 patient samples, the model reveals key clinical findings:

1. **Primary Malignancy Predictors**:
   - **Nuclear Concavity & Concave Points**: Indentations and inward-curving notches along the nuclear boundary serve as the single most critical structural marker differentiating malignant nuclei from benign cells.
   - **Nuclear Size (Radius, Area, Perimeter)**: Malignant cell nuclei demonstrate significantly elevated mean radii and surface area compared to benign specimens due to increased metabolic activity and cellular enlargement.
   - **Texture Variability**: Standard deviation of gray-scale intensity is consistently higher in malignant tumors, reflecting irregular chromatin clumping.

2. **Classification Output**:
   - The model computes a calibrated probability output ($0\%$ to $100\%$) indicating the likelihood of malignancy for a given set of tumor cell measurements.

---

## Key Features

- **Real-Time Browser Inference**: Client-side parameter evaluation providing zero-latency classification updates as sliders change.
- **Interactive Parameter Controls**: Adjust primary parameters (radius, texture, perimeter, area, smoothness, compactness, concavity) and advanced standard error (SE) parameters.
- **Sample Presets**: Quick-load buttons for **Most Benign Sample**, **Average Sample**, and **Most Malignant Sample**.
- **Dataset Visualisations**: Integrated Matplotlib analysis charts highlighting feature distributions, box-plot comparisons, and decision importances.
- **Model Evaluation Output**: Detailed performance reporting featuring highest fold metrics and 5-fold Stratified K-Fold cross-validation results.

---

## Model Performance

Trained using **XGBClassifier** with 5-Fold Stratified K-Fold Cross-Validation:

### Highest Metrics Across Folds
- **Accuracy**: `0.991228` (99.12%)
- **ROC-AUC**: `1.000000` (100.0%)
- **Precision**: `1.000000` (100.0%)
- **Recall**: `1.000000` (100.0%)
- **F1 Score**: `0.988235` (98.82%)

### Mean Metrics Across Folds
- **Accuracy**: `0.978932` (97.89%)
- **ROC-AUC**: `0.994384` (99.44%)
- **Precision**: `0.980783` (98.08%)
- **Recall**: `0.962237` (96.22%)
- **F1 Score**: `0.970881` (97.09%)

---

## Repository Structure

```
.
├── main.py                # XGBoost training pipeline & cross-validation script
├── train_and_export.py    # Feature extraction & model export script
├── index.html             # Main web dashboard interface
├── styles.css             # Styling rules for clean layout
├── app.js                 # Realtime client-side inference engine & UI logic
├── data.csv               # Wisconsin Breast Cancer Diagnostic Dataset
├── model_config.json      # Model coefficients, stats, and sample presets
├── vercel.json            # Deployment configuration for Vercel
└── charts/                # Generated dataset distribution & comparison charts
```

---

## Local Setup & Running

1. **Clone the repository**:
   ```bash
   git clone https://github.com/S-Srinivasan-06/Breast-Cancer-Detection-model.git
   cd Breast-Cancer-Detection-model
   ```

2. **Run locally using Python's built-in HTTP server**:
   ```bash
   python -m http.server 8000
   ```
   Open **`http://localhost:8000`** in your browser.

3. **To retrain the Python model**:
   ```bash
   python main.py
   ```

---

## License & Acknowledgments

- **Dataset**: UCI Machine Learning Repository / Kaggle Wisconsin Breast Cancer Diagnostic.
- **Disclaimer**: For educational & demonstration purposes only. Not intended for clinical diagnostic use.

</div>
