from ultralytics import YOLO

model = YOLO('./data/models/yolov8m.pt')

model.train(data='./data/dataset_fv', 
            epochs=200, 
            batch=32, 
            imgsz=640, 
            cache=True,
            workers=32,
            project="./data/outputs",
            name="fv_mid_exp_v1",
            optimizer="Adam",
            cos_lr=True, 
            close_mosaic=100,
            lr0=0.1, 
            lrf=0.001, 
            dropout=0.1,
            val=True,
            degrees=50,
            flipud=0.5,
            mixup=0.2,
            copy_paste=0.2
            )

metrics = model.val()

model.export(format='onnx')
