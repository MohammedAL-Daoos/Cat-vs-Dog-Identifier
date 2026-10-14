import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. تحميل البيانات المعالجة
loaded = np.load("processed_data.npz")
X = loaded['X']
y = loaded['y']

# 2. تقسيم البيانات (80% تدريب، 20% اختبار)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. إنشاء وتدريب النموذج
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. تقييم كفاءة النموذج الأولي
y_pred = model.predict(X_test)
print(f"دقة النموذج على بيانات التقييم: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nتقرير الأداء التفصيلي:\n", classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))

# 5. حفظ النموذج المدرب
joblib.dump(model, "cat_dog_model.pkl")
print("تم حفظ النموذج بنجاح باسم cat_dog_model.pkl")