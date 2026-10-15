import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

import os
from pathlib import Path

# データパスの設定 (スクリプトと同じディレクトリの data/ フォルダ)
DATA_DIR = Path(__file__).parent / "data"
X = np.load(DATA_DIR / 'superpixel_features.npy')
y = np.load(DATA_DIR / 'superpixel_labels.npy')
print(f"Data: {X.shape}, class distribution: {np.bincount(y)}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
rf = RandomForestClassifier(
    n_estimators=150, max_depth=15, class_weight='balanced',
    n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['keep','invert']))

# 特徴量重要度（トップ10）
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:10]
print("\nTop 10 feature importances:")
for i in indices:
    print(f"  Feature {i}: {importances[i]:.4f}")

joblib.dump(rf, 'dark_invert_rf.pkl')
print("Model saved as dark_invert_rf.pkl")