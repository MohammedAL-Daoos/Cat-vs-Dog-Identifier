# -*- coding: utf-8 -*-
import cv2, os, numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'mobilenetv2.onnx')

if not os.path.exists(model_path):
    print('Error: mobilenetv2.onnx not found!')
    exit(1)

net = cv2.dnn.readNetFromONNX(model_path)
print('Extracting deep features using MobileNetV2...')

def get_deep_features(img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None: return None
    blob = cv2.dnn.blobFromImage(img, 1.0/255.0, (224, 224), (0.485*255, 0.456*255, 0.406*255), swapRB=True, crop=False)
    blob[0, 0, :, :] /= 0.229
    blob[0, 1, :, :] /= 0.224
    blob[0, 2, :, :] /= 0.225
    net.setInput(blob)
    return net.forward().flatten()

X, y = [], []
cats_dir = os.path.join(base_dir, 'dataset', 'cats')
dogs_dir = os.path.join(base_dir, 'dataset', 'dogs')

for f in os.listdir(cats_dir):
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        feat = get_deep_features(os.path.join(cats_dir, f))
        if feat is not None:
            X.append(feat)
            y.append(0)

for f in os.listdir(dogs_dir):
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        feat = get_deep_features(os.path.join(dogs_dir, f))
        if feat is not None:
            X.append(feat)
            y.append(1)

X = np.array(X)
y = np.array(y)
out_file = os.path.join(base_dir, 'processed_data_deep.npz')
np.savez(out_file, X=X, y=y)
print(f'Done! Processed {len(X)} images saved in: {out_file}')
