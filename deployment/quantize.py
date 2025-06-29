from onnxruntime.quantization import quantize_dynamic, QuantType
import os

models_to_quantize = [
    ("models/en_car_plate.onnx", "models/en_car_plate_quant.onnx"),
    ("models/ar_detect_car_number_plate.onnx", "models/ar_detect_car_number_plate_quant.onnx")
]

for model_fp32, model_int8 in models_to_quantize:
    if os.path.exists(model_fp32):
        quantize_dynamic(
            model_input=model_fp32,
            model_output=model_int8,
            weight_type=QuantType.QInt8
        )
        print(f" Quantized {model_fp32} → {model_int8}")
    else:
        print(f" File not found: {model_fp32}")
