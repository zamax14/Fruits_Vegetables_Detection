import os
import shutil
from sklearn.model_selection import train_test_split

dataset_path = "data/fruits_and_vegetables"
images_path = os.path.join(dataset_path, "images")
labels_path = os.path.join(dataset_path, "labels")

output_dir = "data/fruits_and_vegetables"
train_images_dir = os.path.join(output_dir, "images/train")
val_images_dir = os.path.join(output_dir, "images/val")
test_images_dir = os.path.join(output_dir, "images/test")
train_labels_dir = os.path.join(output_dir, "labels/train")
val_labels_dir = os.path.join(output_dir, "labels/val")
test_labels_dir = os.path.join(output_dir, "labels/test")

images = sorted([f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))])
labels = sorted([f for f in os.listdir(labels_path) if f.endswith('.txt')])

print(len(images))
print(len(labels))

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