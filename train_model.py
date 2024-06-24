from ultralytics import YOLO

model = YOLO('data/models/yolov8m.pt')

model.train(data='data/fruits_and_vegetables/data.yaml', epochs=100, batch=8, imgsz=640, name='FV_model')

metrics = model.val()

model.export(format='onnx')
