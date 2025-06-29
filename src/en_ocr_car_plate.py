from ultralytics import YOLO
import easyocr
import cv2

 
model_plate = YOLO("models/en_car_plate.pt")  

 
reader = easyocr.Reader(['en'], gpu=False)

 
cap = cv2.VideoCapture("data/mycarplate.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    for res in model_plate.predict(source=frame, device="cpu", imgsz=640, stream=True):
        for box in res.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            plate = frame[y1:y2, x1:x2]

             
            ocr_res = reader.readtext(plate, detail=1, paragraph=False)
            text = ''.join([t for _, t, conf in ocr_res if conf > 0.4])

             
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

    cv2.imshow("Plate Detection + OCR", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
