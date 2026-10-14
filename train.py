import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# =========================================================
# 1. تحميل مصفوفة الميزات X ومتجه الفئات y من ملف الصور
# =========================================================
data_file = 'processed_data.npz'

if not os.path.exists(data_file):
  print(f"❌ لم يتم العثور على ملف البيانات {data_file}!")
  print('يرجى تشغيل سكربت استخراج السمات أولاً.')
  exit()

# تحميل المصفوفات
loaded = np.load(data_file)
X = loaded['X']  # مصفوفة الميزات (عدد الصور × عدد السمات)
y = loaded['y']  # مصفوفة التسميات (0 للقطط، 1 للكلاب)

print(f'✅ تم تحميل المصفوفات بنجاح:')
print(f'   - شكل مصفوفة الميزات X: {X.shape}')
print(f'   - شكل متجه الفئات y: {y.shape}\n')

# =========================================================
# 2. تقسيم البيانات إلى تدريب واختبار
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# معايرة الميزات (Feature Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# 3. تعريف الخوارزميات المراد المقارنة بينها
# =========================================================
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine (SVM)': SVC(kernel='rbf', probability=True),
    'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Quadratic Discriminant Analysis (QDA)': QuadraticDiscriminantAnalysis(),
}

# =========================================================
# 4. تدريب الخوارزميات وحساب معايير الأداء
# =========================================================
results = []

for name, model in models.items():
  # التدريب باستخدام مصفوفة الميزات
  model.fit(X_train_scaled, y_train)

  # التنبؤ على بيانات الاختبار
  y_pred = model.predict(X_test_scaled)

  # حساب مقاييس الأداء
  acc = accuracy_score(y_test, y_pred)
  prec = precision_score(y_test, y_pred, zero_division=0)
  rec = recall_score(y_test, y_pred, zero_division=0)
  f1 = f1_score(y_test, y_pred, zero_division=0)

  results.append({
      'Algorithm': name,
      'Accuracy': acc,
      'Precision': prec,
      'Recall': rec,
      'F1-Score': f1,
  })

# =========================================================
# 5. عرض جدول المقارنة والنتيجة
# =========================================================
df_results = pd.DataFrame(results).sort_values(
    by='Accuracy', ascending=False
)

print('=================== جدول مقارنة أداء الخوارزميات ===================')
print(df_results.to_string(index=False))

best_model_name = df_results.iloc[0]['Algorithm']
best_accuracy = df_results.iloc[0]['Accuracy']
print(
    f'\n🏆 أفضل خوارزمية لمصفوفة صور القطط والكلاب هي: {best_model_name} بدقة:'
    f' {best_accuracy:.2%}'
)

# =========================================================
# 6. رسم بياني للمقارنة
# =========================================================
plt.figure(figsize=(10, 5))
bars = plt.barh(df_results['Algorithm'], df_results['Accuracy'], color='#3b82f6')
plt.xlabel('Accuracy')
plt.title('Algorithm Comparison - Cats vs Dogs Classification')
plt.xlim(0, 1.05)
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.1%}', ha='left', va='center')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('comparison_chart.png', dpi=300)
print('📊 تم حفظ الرسم البياني للمقارنة في: comparison_chart.png')