import os
import yaml

current_dir = os.getcwd()

data = {
    'train': os.path.join(current_dir, 'data/fruits_and_vegetables/train'),
    'val': os.path.join(current_dir, 'data/fruits_and_vegetables/val'),
    'nc': 18,
    'names': [
        'aguacate', 
        'brocoli', 
        'calabacita', 
        'cebolla', 
        'chile', 
        'fresa', 
        'jitomate', 
        'lechuga', 
        'limon', 
        'mango', 
        'manzana', 
        'naranja', 
        'papa', 
        'pepino', 
        'pinia', 
        'platano', 
        'sandia', 
        'zanahoria'
    ]
}

with open('data/data.yaml', 'w') as file:
    yaml.dump(data, file, default_flow_style=False)
