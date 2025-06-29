# 🚘 Car Plate Recognition

This project provides a **real-time   car plate recognition system**, supporting detection and recognition of Arabic letters and digits using **YOLOv8** and **OCR techniques**.

---

## 🎯 Project Objectives

- Detect car plates in images and video using object detection.
- Extract and recognize Arabic characters using two different approaches:
  1. **YOLOv8-based character detection**.
  2. **OCR-based character recognition**.
- Compare and benchmark both approaches for accuracy and performance.
- Optimize models for deployment using **ONNX** and **INT8 quantization**.

---

## 🧠 Model Approaches

### 🔹 Approach 1: YOLO + YOLO
- **First model** detects the plate location.
- **Second model** detects Arabic digits/letters inside the plate.
- Both models are trained and exported to ONNX.

### 🔹 Approach 2: YOLO + OCR
- Plate is detected using YOLO.
- Text is extracted using OCR (tested on Arabic plates).

---

## ⚙️ Tech Stack

| Tool           | Purpose                              |
|----------------|--------------------------------------|
| YOLOv8         | Plate and character detection        |
| EasyOCR        |  OCR recognition       |
| ONNX           | Model export & optimization          |
| ONNX Runtime   | Fast inference on exported models    |
| OpenCV         | Image preprocessing & visualization  |
| Python         | Full implementation                  |

