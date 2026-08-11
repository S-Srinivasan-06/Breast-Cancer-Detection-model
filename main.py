import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data.csv")

df = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])

df['area_to_perimeter'] = df['area_mean'] / df['perimeter_mean']
df['circularity'] = df['area_mean'] / (df['radius_mean']**2)
df['radius_growth'] = df['radius_worst'] - df['radius_mean']
df['texture_variance'] = df['texture_worst'] - df['texture_mean']
df['concavity_density'] = df['concavity_mean'] / df['concave points_mean']

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
scale_weight = np.sum(y==0) / np.sum(y==1)

# Updated with optimized hyperparameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'scale_pos_weight': scale_weight,
    'tree_method': 'hist',
    'random_state': 10,
    'n_estimators': 746,
    'learning_rate': 0.061201,
    'max_depth': 4,
    'subsample': 0.664585,
    'colsample_bytree': 0.805884,
    'min_child_weight': 2,
    'gamma': 0.000084,
    'reg_alpha': 0.115272,
    'reg_lambda': 0.002050,
}

metrics = []
oof_preds = np.zeros(len(y))
oof_probas = np.zeros(len(y))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False) 
        
    preds = model.predict(X_val)
    preds_proba = model.predict_proba(X_val)[:, 1]
    
    oof_preds[val_idx] = preds
    oof_probas[val_idx] = preds_proba
    
    accuracy = accuracy_score(y_val, preds)
    auc = roc_auc_score(y_val, preds_proba)
    precision = precision_score(y_val, preds)
    recall = recall_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    
    metrics.append({
        'accuracy': accuracy,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

avg_accuracy = np.mean([m['accuracy'] for m in metrics])
avg_auc = np.mean([m['auc'] for m in metrics])

print("\n=== Out-of-Fold Classification Report ===")
print(classification_report(y, oof_preds, target_names=le.classes_))
print(f"Overall OOF ROC-AUC: {roc_auc_score(y, oof_probas):.4f}")

results_df = pd.DataFrame(metrics)
print("\n=== Mean Metrics Across Folds ===")
print(results_df.mean())