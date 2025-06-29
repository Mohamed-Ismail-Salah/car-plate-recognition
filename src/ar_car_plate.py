from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

model_plate = YOLO("models/ar_car_plate_model.pt")
model_char = YOLO("models/ar_detect_car_number_plate.pt")

mapping = {
    '1':'١','2':'٢','3':'٣','4':'٤','5':'٥',
    '6':'٦','7':'٧','8':'٨','9':'٩',
    'taa':'ط','ain':'ع','alif':'ا','baa':'ب',
    'daal':'د','faa':'ف','haa':'ه','jeem':'ج',
    'laam':'ل','meem':'م','noon':'ن','qaaf':'ق',
    'raa':'ر','saad':'ص','seen':'س','waw':'و','yaa':'ي'
}

font_path = "arial.ttf"   
font_size = 30

 
def put_arabic_text(cv2_img, text, pos):
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(pos, text, fill=(0,255,0), font=font)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

 
cap = cv2.VideoCapture("data/arabic_number_plate_recognition.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

  
    for res_plate in model_plate.predict(source=frame, device='cpu', imgsz=640, stream=True,):
        for box in res_plate.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]

 
            for res_char in model_char.predict(source=crop, device='cpu', imgsz=320, stream=True):
                chars = sorted(res_char.boxes, key=lambda b: b.xyxy[0][0].item())
                confs = [b.conf for b in chars]
 
                text = ' '.join(
                    mapping[model_char.names[int(b.cls)]]
                    for b, conf in zip(chars, confs)
                    if model_char.names[int(b.cls)] in mapping and conf > 0.4
                )

 
                frame = put_arabic_text(frame, text, (x1, y1 - 15))

 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

 
    cv2.imshow("Plate + Arabic OCR", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
