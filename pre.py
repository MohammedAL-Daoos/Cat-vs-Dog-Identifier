
import cv2
import numpy as np
import os

def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

data = []
labels = []

# تحديد المسار المطلق لضمان العثور على المجلد دائماً
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, "dataset")
#هنا ترميز للكلاسين الموجودين 
categories = {"cats": 0, "dogs": 1}

# امتدادات الصور المقبولة
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

print(f"جاري البحث عن الصور في المسار: {dataset_path}")

for category, label in categories.items():
    folder_path = os.path.join(dataset_path, category)
    
    if not os.path.exists(folder_path):
        print(f"⚠️ تحذير: المجلد غير موجود -> {folder_path}")
        continue
        
    count = 0
    for img_name in os.listdir(folder_path):
        # التحقق من امتداد الملف
        if not img_name.lower().endswith(VALID_EXTENSIONS):
            continue
            
        img_path = os.path.join(folder_path, img_name)
        # استخدام imdecode لدعم المسارات التي تحتوي على أحرف عربية في ويندوز
        try:
            image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            image = None
        
        if image is None:
            print(f"⚠️ تعذر قراءة الصورة: {img_path}")
            continue
            
        image = cv2.resize(image, (128, 128))
        features = extract_color_histogram(image)
        
        data.append(features)
        labels.append(label)
        count += 1
        
    print(f" تم تحميل {count} صورة من صنف '{category}'")

# التأكد من وجود بيانات قبل الحفظ
if len(data) == 0:
    print("\n❌ خطأ: لم يتم العثور على أي صورة! تأكد من وجود الصور داخل مجلدات cats و dogs.")
else:
    X = np.array(data)
    y = np.array(labels)
    
    output_file = os.path.join(base_dir, "processed_data.npz")
    np.savez(output_file, X=X, y=y)
    print(f"\n✅ تم استخراج السمات بنجاح! إجمالي الصور: {len(X)}")
    print(f"تم حفظ الملف في: {output_file}")