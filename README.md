🐱🐶 Cats vs Dogs Identifier

مشروع في الرؤية الحاسوبية وتعلّم الآلة (Computer Vision & Machine Learning) لتصنيف صور القطط والكلاب باستخدام نهجين: تعلّم الآلة التقليدي واستخراج السمات العميقة باستخدام MobileNetV2.

يتضمن المشروع مقارنة بين عدة خوارزميات تصنيف، بالإضافة إلى واجهة رسومية تفاعلية تتيح للمستخدم اختبار الصور ومقارنة نتائج النموذجين.

Key Features

- مقارنة أداء خوارزميات تعلّم الآلة التقليدية والتعلّم العميق.
- استخدام HSV Color Histograms وColor Moments لاستخراج السمات اللونية.
- استخدام MobileNetV2 لاستخراج السمات العميقة.
- واجهة رسومية للتنبؤ بفئة الصورة وعرض نسبة الثقة.
- نماذج مدرّبة وجاهزة للاستخدام.

Results

حقق النموذج التقليدي دقة 63.11%، بينما حقق النموذج المعتمد على السمات العميقة دقة 98.54% وفقًا للنتائج المسجلة في المشروع.

Technologies

Python · OpenCV · MobileNetV2 · Scikit-learn · Machine Learning · Deep Learning

Run the Project

pip install -r requirements.txt
python app_gui.py

Project Type: Academic AI & Computer Vision Project
