from ultralytics import YOLO


model = YOLO("./data/models/best.pt")

results = model(["im1.jpg", "im2.jpg"])

for result in results:
    result.show() 