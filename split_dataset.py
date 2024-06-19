import os
import shutil
from sklearn.model_selection import train_test_split

dataset_path = "path/to/your/yolo/dataset"
images_path = os.path.join(dataset_path, "images")
labels_path = os.path.join(dataset_path, "labels")

output_dir = "path/to/output/dataset"
train_images_dir = os.path.join(output_dir, "train/images")
val_images_dir = os.path.join(output_dir, "val/images")
test_images_dir = os.path.join(output_dir, "test/images")
train_labels_dir = os.path.join(output_dir, "train/labels")
val_labels_dir = os.path.join(output_dir, "val/labels")
test_labels_dir = os.path.join(output_dir, "test/labels")


images = sorted([f for f in os.listdir(images_path) if f.endswith('.jpg')])
labels = sorted([f for f in os.listdir(labels_path) if f.endswith('.txt')])

images = [f.replace('.txt', '.jpg') for f in labels]

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.25, random_state=42)

def move_files(files, src_dir, dst_dir):
    for f in files:
        shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

move_files(train_images, images_path, train_images_dir)
move_files(val_images, images_path, val_images_dir)
move_files(test_images, images_path, test_images_dir)

move_files(train_labels, labels_path, train_labels_dir)
move_files(val_labels, labels_path, val_labels_dir)
move_files(test_labels, labels_path, test_labels_dir)

print("Dataset split completed!")