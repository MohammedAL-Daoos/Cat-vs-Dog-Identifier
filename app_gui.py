# -*- coding: utf-8 -*-
import os, sys
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import joblib
from PIL import Image, ImageTk

# ضبط مسار العمل على مجلد الملف الحالي لتفادي مشاكل المسارات العربية في OpenCV
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# 1. تحميل النموذج الكلاسيكي (الدكتور)
try:
    classic_model = joblib.load('cat_dog_model.pkl') if os.path.exists('cat_dog_model.pkl') else None
except Exception:
    classic_model = None

# 2. تحميل نموذج التعلم العميق المتطور (98.5%)
try:
    deep_net = cv2.dnn.readNetFromONNX('mobilenetv2.onnx') if os.path.exists('mobilenetv2.onnx') else None
    deep_model = joblib.load('cat_dog_deep_model.pkl') if os.path.exists('cat_dog_deep_model.pkl') else None
    deep_scaler = joblib.load('cat_dog_deep_scaler.pkl') if os.path.exists('cat_dog_deep_scaler.pkl') else None
except Exception as e:
    deep_net = None
    deep_model = None
    deep_scaler = None

def extract_classic_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, (8, 8, 8), [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().reshape(1, -1)

def extract_deep_features(image):
    blob = cv2.dnn.blobFromImage(image, 1.0/255.0, (224, 224), (0.485*255, 0.456*255, 0.406*255), swapRB=True, crop=False)
    blob[0, 0, :, :] /= 0.229
    blob[0, 1, :, :] /= 0.224
    blob[0, 2, :, :] /= 0.225
    deep_net.setInput(blob)
    feats = deep_net.forward().flatten().reshape(1, -1)
    return deep_scaler.transform(feats)

root = tk.Tk()
root.title('🐱🐶 نظام فحص وتصنيف صور القطط والكلاب')
root.geometry('600x720')
root.configure(bg='#0f172a')
root.resizable(False, False)

# Header
header = tk.Frame(root, bg='#1e293b', pady=12)
header.pack(fill='x')
tk.Label(header, text='🐱🐶 فاحص صور القطط والكلاب الذكي', font=('Segoe UI', 17, 'bold'), bg='#1e293b', fg='#f8fafc').pack()
tk.Label(header, text='قارن بين تعلم الآلة الكلاسيكي والتعلم العميق فائق الدقة', font=('Segoe UI', 9), bg='#1e293b', fg='#94a3b8').pack(pady=2)

main_frame = tk.Frame(root, bg='#0f172a', padx=20, pady=10)
main_frame.pack(fill='both', expand=True)

# Radio buttons to choose model mode
mode_var = tk.StringVar(value='deep')

mode_frame = tk.Frame(main_frame, bg='#1e293b', padx=10, pady=8, bd=1, relief='ridge')
mode_frame.pack(fill='x', pady=5)

tk.Label(mode_frame, text='اختر المحرك المستخدم في الفحص:', font=('Segoe UI', 10, 'bold'), bg='#1e293b', fg='#e2e8f0').pack(anchor='w', padx=5)

rb_deep = tk.Radiobutton(
    mode_frame,
    text='⚡ نموذج التعلم العميق (MobileNetV2) - دقة فائقة 98.5% 🔥',
    variable=mode_var,
    value='deep',
    font=('Segoe UI', 10, 'bold'),
    bg='#1e293b',
    fg='#10b981',
    selectcolor='#0f172a',
    activebackground='#1e293b',
    activeforeground='#10b981'
)
rb_deep.pack(anchor='w', padx=5, pady=2)

rb_classic = tk.Radiobutton(
    mode_frame,
    text='📚 نموذج الدكتور الكلاسيكي (Color Histogram) - دقة 63.1%',
    variable=mode_var,
    value='classic',
    font=('Segoe UI', 9),
    bg='#1e293b',
    fg='#cbd5e1',
    selectcolor='#0f172a',
    activebackground='#1e293b',
    activeforeground='#cbd5e1'
)
rb_classic.pack(anchor='w', padx=5, pady=2)

# Image Card
img_card = tk.Frame(main_frame, bg='#1e293b', width=320, height=320, bd=1, relief='flat')
img_card.pack(pady=10)
img_card.pack_propagate(False)

img_label = tk.Label(img_card, text='🖼️\n\nلم يتم اختيار صورة بعد\nاضغط على الزر بالأسفل لاختيار صورة', font=('Segoe UI', 11), bg='#1e293b', fg='#64748b', justify='center')
img_label.pack(expand=True)

current_image_path = None

def run_prediction(img_path):
    global current_image_path
    current_image_path = img_path
    try:
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        messagebox.showerror('خطأ', f'تعذر قراءة الصورة: {e}')
        return

    if image is None:
        messagebox.showerror('خطأ', 'تعذر قراءة الصورة!')
        return

    mode = mode_var.get()
    if mode == 'deep' and deep_model is not None and deep_net is not None:
        feats = extract_deep_features(image)
        pred = deep_model.predict(feats)[0]
        probs = deep_model.predict_proba(feats)[0]
        engine_text = 'المحرك: التعلم العميق (MobileNetV2 - 98.5%)'
    else:
        resized = cv2.resize(image, (128, 128))
        feats = extract_classic_features(resized)
        pred = classic_model.predict(feats)[0]
        probs = classic_model.predict_proba(feats)[0]
        engine_text = 'المحرك: تعلم الآلة الكلاسيكي (Color Histogram - 63.1%)'

    confidence = probs[pred] * 100
    if pred == 0:
        label = 'قطة (Cat) 🐱'
        color = '#38bdf8'
    else:
        label = 'كلب (Dog) 🐶'
        color = '#fb923c'

    result_badge.config(text=label, fg=color)
    conf_badge.config(text=f'نسبة التأكد (Confidence): {confidence:.1f}%')
    engine_label.config(text=engine_text)
    filename_label.config(text=f'📄 {os.path.basename(img_path)}')

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    pil_img.thumbnail((300, 300))
    tk_img = ImageTk.PhotoImage(pil_img)
    img_label.config(image=tk_img, text='')
    img_label.image = tk_img

def recheck_mode():
    if current_image_path:
        run_prediction(current_image_path)

rb_deep.config(command=recheck_mode)
rb_classic.config(command=recheck_mode)

def choose_file():
    path = filedialog.askopenfilename(
        title='اختر صورة قطة أو كلب من جهازك',
        initialdir=os.path.join(BASE_DIR, 'test') if os.path.exists(os.path.join(BASE_DIR, 'test')) else BASE_DIR,
        filetypes=[('Images', '*.jpg *.jpeg *.png *.bmp *.webp'), ('All Files', '*.*')]
    )
    if path:
        run_prediction(path)

# Result Card
res_card = tk.Frame(main_frame, bg='#1e293b', padx=15, pady=8)
res_card.pack(fill='x', pady=5)

result_badge = tk.Label(res_card, text='بانتظار اختيار الصورة...', font=('Segoe UI', 15, 'bold'), bg='#1e293b', fg='#94a3b8')
result_badge.pack()

conf_badge = tk.Label(res_card, text='', font=('Segoe UI', 11, 'bold'), bg='#1e293b', fg='#cbd5e1')
conf_badge.pack()

engine_label = tk.Label(res_card, text='', font=('Segoe UI', 9), bg='#1e293b', fg='#10b981')
engine_label.pack(pady=1)

filename_label = tk.Label(res_card, text='', font=('Segoe UI', 8), bg='#1e293b', fg='#64748b')
filename_label.pack()

# Browse Button
browse_btn = tk.Button(
    main_frame,
    text='📂 اضغط هنا لاختيار صورة من جهازك',
    font=('Segoe UI', 12, 'bold'),
    bg='#2563eb',
    fg='white',
    activebackground='#1d4ed8',
    activeforeground='white',
    relief='flat',
    padx=20,
    pady=8,
    cursor='hand2',
    command=choose_file
)
browse_btn.pack(pady=8)

root.mainloop()
