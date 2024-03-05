import os
import random
import cv2 as cv
from src.shape_detector import predict
from src.isolate_character import isolate_character, isolate_character_exp
from src.color_extractor import get_shape_color
import numpy as np

TESSERACT_CUSTOM_CONFIG = r'--psm 10'

def add_bound_box(src_img, xywh_tensor_values, label): 
    bbox = []
    for i in range(len(xywh_tensor_values)):
        bbox.append(int(xywh_tensor_values[i]))
    top_left = (bbox[0] - int(bbox[2]/2), bbox[1] - int(bbox[3]/2))
    bottom_right = (top_left[0] + bbox[2], top_left[1] + bbox[3])
    cv.rectangle(src_img, top_left, bottom_right, (0, 255, 0), 1)
    (text_width, text_height), baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    text_origin = (top_left[0], top_left[1] - 10)
    cv.rectangle(src_img, text_origin, (text_origin[0] + text_width, text_origin[1] - text_height - baseline), (0, 255, 0), thickness=cv.FILLED)
    cv.putText(src_img, label, text_origin, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return src_img


def get_random_img():
    dir = os.path.abspath("standard_object_dataset/test/images") 
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    return os.path.abspath(f"standard_object_dataset/test/images/{random.choice(files)}")


def crop_image(src_image, xywh):
    for i in range(len(xywh)):
        xywh[i] = int(xywh[i])
    w = xywh[2] - 10 
    h = xywh[3] - 10 
    x = xywh[0] - int(w/2)
    y = xywh[1] - int(h/2) 
    bbox = np.array([x, y, w, h])
    return src_image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]


def debug():
    DEFAULT_SHAPE_DETECT_MODEL_PATH = "runs/detect/exp/weights/best.pt"

    random_img = get_random_img()
    img = cv.imread(random_img)

    res = predict(DEFAULT_SHAPE_DETECT_MODEL_PATH, random_img)
    tensor = res[0].boxes.xywh.clone().detach()
    try:
        xywh_tensor_values = tensor.cpu().numpy().tolist()[0] 
    except:
        return
    cropped_img = crop_image(img, xywh_tensor_values)

    #isolated_character_image = isolate_character(cropped_img) 
    isolated_character_image = isolate_character_exp(cropped_img) 

    classes = res[0].names
    box_class = int(res[0].boxes.cls.cpu().numpy().tolist()[0])
    label = classes[box_class]
    image_with_box = add_bound_box(img, xywh_tensor_values, label)

    cv.imshow("shape detection", image_with_box)
    cv.imshow("isolated character", isolated_character_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    while True:
        debug() 
