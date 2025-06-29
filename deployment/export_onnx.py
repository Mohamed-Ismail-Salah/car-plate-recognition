from ultralytics import YOLO

plate_model = YOLO("models/en_car_plate.pt")
plate_model.export(
    format="onnx",
    imgsz=640,
    simplify=True,
    opset=12,
    dynamic=False
)
print(" Plate model exported to ONNX.")

char_model = YOLO("models/ar_detect_car_number_plate.pt")
char_model.export(
    format="onnx",
    imgsz=640,
    simplify=True,
    opset=12,
    dynamic=False
)
print(" Character model exported to ONNX.")
