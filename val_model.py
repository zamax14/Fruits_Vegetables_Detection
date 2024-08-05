from ultralytics import YOLO
import json

model = YOLO("data/models/best_v9.pt")

metrics = model.val(data='data/fruits_and_vegetables/data.yaml', save_json=True, batch=1)

data = {"map": metrics.box.map, 
        "map50": metrics.box.map50, 
        "map75": metrics.box.map75, 
        "maps":metrics.box.maps.tolist(),
        "Precision": metrics.box.mp,
        "Recall": metrics.box.mr
}

print(data)

with open("metrics.json", 'w') as outfile:
        json.dump(data, outfile, indent=4)