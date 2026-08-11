import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json

# Load dataset
df = pd.read_csv("data.csv")
df = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Select 10 main mean features for intuitive input
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'
]

X = df[features]
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_train_scaled, y_train)

acc = accuracy_score(y_test, clf.predict(X_test_scaled))
print(f"Test Accuracy: {acc * 100:.2f}%")

# Compute min, max, mean, median for UI sliders
feature_stats = {}
for feat in features:
    feature_stats[feat] = {
        'min': float(df[feat].min()),
        'max': float(df[feat].max()),
        'mean': float(df[feat].mean()),
        'std': float(scaler.scale_[features.index(feat)]),
        'scaler_mean': float(scaler.mean_[features.index(feat)])
    }

# Sample benign and malignant rows for preset buttons
benign_sample = df[df['diagnosis'] == 0][features].iloc[0].to_dict()
malignant_sample = df[df['diagnosis'] == 1][features].iloc[0].to_dict()

model_config = {
    'features': features,
    'accuracy': float(acc),
    'intercept': float(clf.intercept_[0]),
    'coefficients': [float(c) for c in clf.coef_[0]],
    'stats': feature_stats,
    'samples': {
        'benign': benign_sample,
        'malignant': malignant_sample
    }
}

with open("model_config.json", "w") as f:
    json.dump(model_config, f, indent=2)

print("Saved model_config.json successfully!")
