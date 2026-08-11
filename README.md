# Breast Cancer Detection & Classification Model

An interactive, real-time web application and machine learning model for classifying cell nucleus measurements from breast mass aspirates into **Benign** or **Malignant** diagnosis.

Based on the [UCI Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).

---

## 🌟 Key Features

- **Real-Time Browser Inference**: Client-side parameter calculation providing zero-latency classification updates as sliders change.
- **Interactive Parameter Controls**: Adjust primary parameters (radius, texture, perimeter, area, smoothness, compactness, concavity) and advanced standard error (SE) parameters.
- **Sample Presets**: Quick-load buttons for **Most Benign Sample**, **Average Sample**, and **Most Malignant Sample**.
- **Dataset Visualisations**: Integrated Matplotlib analysis charts highlighting feature distributions, box-plot comparisons, and decision importances.
- **Model Evaluation Output**: Detailed performance reporting featuring highest fold metrics and 5-fold Stratified K-Fold cross-validation results.

---

## 📊 Model Performance

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

## 📁 Repository Structure

```
.
├── main.py                # XGBoost training pipeline & cross-validation script
├── train_and_export.py    # Feature extraction & model export script
├── index.html             # Main web dashboard interface
├── styles.css             # Styling rules for clean, responsive layout
├── app.js                 # Realtime client-side inference engine & UI logic
├── data.csv               # Wisconsin Breast Cancer Diagnostic Dataset
├── model_config.json      # Model coefficients, stats, and sample presets
├── vercel.json            # Deployment configuration for Vercel
└── charts/                # Generated dataset distribution & comparison charts
```

---

## 🚀 Local Setup & Running

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

## 🌐 Live Demo Deployment (Vercel)

This repository includes a `vercel.json` configuration for 1-click deployment:
1. Import this repository in [Vercel](https://vercel.com/new).
2. Click **Deploy** to instantly generate your live demo link.

---

## 📜 License & Acknowledgments

- **Dataset**: UCI Machine Learning Repository / Kaggle Wisconsin Breast Cancer Diagnostic.
- **Disclaimer**: For educational & demonstration purposes only. Not intended for clinical diagnostic use.
