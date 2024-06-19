from ultralytics import YOLO

model = YOLO("runs/detect/FV_model4/weights/best.pt")

model.predict("data/fruits_and_vegetables/new_test/Screenshot from 2024-06-19 15-06-24.png", save=True, imgsz=640, conf=0.5)