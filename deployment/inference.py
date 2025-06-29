import time
import cv2
import numpy as np
import onnxruntime as ort

def preprocess(img, size=640):
    img = cv2.resize(img, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]
    return img

def measure(path, iters=50):
    if not path.endswith(".onnx"):
        print(f" Skipping {path} (not ONNX)")
        return
    try:
        sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        img = cv2.imread("test.jpg")
        if img is None:
            raise FileNotFoundError(" Couldn't find test.jpg.")
        inp = preprocess(img)
        sess.run(None, {'images': inp})
        ts = []
        for _ in range(iters):
            t0 = time.time()
            sess.run(None, {'images': inp})
            ts.append(time.time() - t0)
        avg_ms = np.mean(ts) * 1000
        print(f" {path} → {avg_ms:.2f} ms")
    except Exception as e:
        print(f" Failed on {path}: {e}")

 
model_names = [
    "en_car_plate",
    "ar_detect_car_number_plate"
]

 
for name in model_names:
    print(f"\n📌 Measuring: {name}")
    measure(f"models/{name}.onnx")
    measure(f"models/{name}_quant.onnx")
