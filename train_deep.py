import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

data_path = 'processed_data_deep.npz'
if not os.path.exists(data_path):
    print('Error: processed_data_deep.npz not found!')
    sys.exit(1)

data = np.load(data_path)
X, y = data['X'], data['y']
print(f'Deep Features loaded: X shape={X.shape}, y shape={y.shape}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(C=0.1, max_iter=500, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine (SVM)': SVC(kernel='linear', C=0.1, probability=True, random_state=42)
}

results = []
trained_models = {}

print('\n=================== Deep Learning Models Comparison ===================')
for name, m in models.items():
    m.fit(X_train_s, y_train)
    preds = m.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    results.append({'Algorithm': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1})
    trained_models[name] = m

df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
for r in df.to_dict('records'):
    print(f"{r['Algorithm']:<30} Accuracy: {r['Accuracy']:.2%}  Precision: {r['Precision']:.2%}  Recall: {r['Recall']:.2%}  F1: {r['F1-Score']:.2%}")

best_name = df.iloc[0]['Algorithm']
best_acc = df.iloc[0]['Accuracy']
print(f'\nTop Model: {best_name} with Accuracy: {best_acc:.2%}')

# Save the best model and scaler
best_model = trained_models[best_name]
joblib.dump(best_model, 'cat_dog_deep_model.pkl')
joblib.dump(scaler, 'cat_dog_deep_scaler.pkl')
print('Model saved in: cat_dog_deep_model.pkl')
print('Scaler saved in: cat_dog_deep_scaler.pkl')

# Generate Chart
plt.figure(figsize=(10, 5))
bars = plt.barh(df['Algorithm'], df['Accuracy'], color='#10b981')
plt.xlabel('Accuracy')
plt.title('Deep Learning Features (MobileNetV2) - Cats vs Dogs Accuracy')
plt.xlim(0.8, 1.05)
for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.005, bar.get_y() + bar.get_height()/2, f'{w:.2%}', ha='left', va='center', fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('deep_comparison_chart.png', dpi=300)
print('Comparison chart saved: deep_comparison_chart.png')
