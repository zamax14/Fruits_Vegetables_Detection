from ultralytics import YOLO


model = YOLO("data/models/best_v9_9.pt")

results = model('./test/')

for result in results:
    result.save()
    #result.show()

model.export(format='onnx')
