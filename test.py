import cv2
import numpy as np
import joblib

# دالة استخراج السمات (نفس المستخدمة أثناء التدريب)
def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# 1. تحميل النموذج المدرب
model = joblib.load("cat_dog_model.pkl")

# 2. تحميل صورة جديدة للاختبار (عبر نافذة اختيار بالماوس أو عبر مسار)
import os, sys

if len(sys.argv) > 1:
    test_image_path = sys.argv[1]
else:
    # فتح نافذة لاختيار الصورة بالماوس
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        test_dir = os.path.join(curr_dir, "test")
        file_selected = filedialog.askopenfilename(
            title="اختر صورة قطة أو كلب لاختبارها",
            initialdir=test_dir if os.path.exists(test_dir) else curr_dir,
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"), ("All files", "*.*")]
        )
        root.destroy()
        test_image_path = file_selected if file_selected else os.path.join("test", "dog1.jpg")
    except Exception:
        test_image_path = os.path.join("test", "dog1.jpg")

print(f"\n🔍 جاري اختبار الصورة: {test_image_path}")

# قراءة الصورة مع دعم المسارات باللغة العربية
try:
    image = cv2.imdecode(np.fromfile(test_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
except Exception:
    image = None

if image is None:
    print(f"❌ تعذر قراءة الصورة من المسار: {test_image_path}")
else:
    # 3. معالجة الصورة بنفس خطوات التدريب
    resized_img = cv2.resize(image, (128, 128))
    features = extract_color_histogram(resized_img).reshape(1, -1)

    # 4. التنبؤ
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    label_map = {0: "قطة (Cat)", 1: "كلب (Dog)"}
    result = label_map[prediction]
    confidence = probabilities[prediction] * 100

    # 5. عرض النتيجة على الصورة بشكل واضح
    display_img = image.copy()
    h, w = display_img.shape[:2]
    # شريط خلفية داكن في أعلى الصورة لقراءة أوضح
    cv2.rectangle(display_img, (0, 0), (w, 60), (25, 25, 25), -1)
    
    text = f"Result: {result} | Confidence: {confidence:.1f}%"
    cv2.putText(display_img, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 100), 2)
    
    output_result = "test_result.png"
    cv2.imwrite(output_result, display_img)
    print(f"🎯 النتيجة: {result} | نسبة الثقة: {confidence:.2f}%")
    print(f"💾 تم حفظ الصورة مع النتيجة في: {output_result}")
    
    # فتح الصورة فوراً في عارض صور ويندوز
    try:
        os.startfile(output_result)
    except Exception:
        pass