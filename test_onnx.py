import onnxruntime as ort
import cv2
import numpy as np


def preprocess_image(image_path, input_shape):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (input_shape[3], input_shape[2]))
    image = image.astype(np.float32)
    image /= 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

def postprocess_output(outputs, image_shape, input_shape):
    boxes = outputs[0][:, :4]
    scores = outputs[0][:, 4]
    labels = np.argmax(outputs[0][:, :5])
    scale_x = image_shape[1] / input_shape[3]
    scale_y = image_shape[0] / input_shape[2]
    boxes[:, 0::2] *= scale_x
    boxes[:, 1::2] *= scale_y
    return boxes, scores, labels


model_path = 'data/models/best.onnx'
session = ort.InferenceSession(model_path)

input_shape = session.get_inputs()[0].shape
input_name = session.get_inputs()[0].name
image_path = 'img.png'
input_data = preprocess_image(image_path, input_shape)

outputs = session.run(None, {input_name: input_data})


image = cv2.imread(image_path)
image_shape = image.shape
boxes, scores, labels = postprocess_output(outputs, image_shape, input_shape)

for b,s,l in zip(boxes, scores, labels):
    print(b,s,l)

