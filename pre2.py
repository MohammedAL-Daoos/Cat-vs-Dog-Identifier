import os
import cv2
import numpy as np
import pandas as pd


def extract_image_features(image):
  # 1. متوسط الألوان والانحراف المعياري لليناء B, G, R
  mean_b, mean_g, mean_r = cv2.mean(image)[:3]
  std_b, std_g, std_r = cv2.meanStdDev(image)[1].flatten()[:3]

  # 2. تحويل إلى HSV واستخراج الهستغرام اللوني
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  hist = cv2.calcHist([hsv], [0, 1, 2], None, (8, 8, 8), [0, 180, 0, 256, 0, 256])
  cv2.normalize(hist, hist)
  hist_features = hist.flatten()

  # تجميع كافة السمات
  features = [mean_b, mean_g, mean_r, std_b, std_g, std_r] + list(
      hist_features
  )
  return features


dataset_path = 'dataset'
categories = {'cats': 0, 'dogs': 1}
records = []

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

for category, label in categories.items():
  folder_path = os.path.join(dataset_path, category)
  if not os.path.exists(folder_path):
    continue

  for img_name in os.listdir(folder_path):
    if not img_name.lower().endswith(VALID_EXTENSIONS):
      continue

    img_path = os.path.join(folder_path, img_name)
    try:
      image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
      image = None

    if image is None:
      continue

    image = cv2.resize(image, (128, 128))
    feats = extract_image_features(image)

    # تجهيز السجل لإضافته للجدول
    row = {
        'Image_Name': img_name,
        'Category': category,
        'Label': label,
        'Mean_Blue': round(feats[0], 2),
        'Mean_Green': round(feats[1], 2),
        'Mean_Red': round(feats[2], 2),
        'Std_Blue': round(feats[3], 2),
        'Std_Green': round(feats[4], 2),
        'Std_Red': round(feats[5], 2),
    }

    # إضافة بقية سمات الهستغرام
    for idx, hist_val in enumerate(feats[6:]):
      row[f'Hist_Bin_{idx+1}'] = round(hist_val, 5)

    records.append(row)

# تحويل البيانات إلى DataFrame وحفظها كملف Excel
df = pd.DataFrame(records)
excel_path = 'extracted_features.xlsx'
df.to_excel(excel_path, index=False)

print(
    f'✅ تم استخراج السمات لـ {len(df)} صورة وحفظها في ملف الاكسل:'
    f' {excel_path}'
)