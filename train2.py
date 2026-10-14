import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. قراءة البيانات من ملف Excel
df = pd.read_excel('extracted_features.xlsx')

# 2. فصل سمات الصور (X) والتصنيف (y)
# استبعاد أعمدة اسم الصورة والتصنيف النصي
X = df.drop(columns=['Image_Name', 'Category', 'Label']).values
y = df['Label'].values

# 3. تقسيم البيانات لتدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. المعايرة المعيارية
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. تدريب ومقارنة الخوارزميات
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine (SVM)': SVC(kernel='rbf', random_state=42),
    'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    results.append({
        'Algorithm': name,
        'Accuracy': f"{accuracy_score(y_test, y_pred):.2%}",
        'Precision': f"{precision_score(y_test, y_pred, zero_division=0):.2%}",
        'Recall': f"{recall_score(y_test, y_pred, zero_division=0):.2%}",
        'F1-Score': f"{f1_score(y_test, y_pred, zero_division=0):.2%}"
    })

# 6. عرض النتائج
df_results = pd.DataFrame(results)
print("=== نتائج المقارنة بناءً على سمات ملف Excel ===")
print(df_results.to_string(index=False))