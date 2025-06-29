from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display


model_plate = YOLO("models/ar_car_plate_model.pt")

reader = easyocr.Reader(['ar'], gpu=False)

font = ImageFont.truetype("arial.ttf", 32)


def draw_arabic(frame, text, pos, color=(0,255,0)):
    reshaped = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped)
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(pos, bidi_text, fill=color, font=font)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


cap = cv2.VideoCapture("data/arabic_number_plate_recognition.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    for res in model_plate.predict(source=frame, device="cpu", imgsz=640, stream=True):
        for box in res.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            plate_crop = frame[y1:y2, x1:x2]

             
            ocr_results = reader.readtext(
                plate_crop, 
                allowlist='ابتثجحخدذرزسشصضطظعغفقكلمنهوي٠١٢٣٤٥٦٧٨٩',
                detail=1, 
                paragraph=False
            )
            
            plate_text = ''.join([t for bbox, t, conf in ocr_results if conf > 0.4])

             
            frame = draw_arabic(frame, plate_text, (x1, y1 - 40))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("Plate Detection + Arabic OCR", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
