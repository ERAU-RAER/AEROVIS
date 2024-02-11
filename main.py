import os
import random
import cv2 as cv
from src.shape_detector import predict
from src.isolate_character import isolate_character, crop_image

def get_random_img():
    dir = os.path.abspath("standard_object_shape-1/test/images") 
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    return os.path.abspath(f"standard_object_shape-1/test/images/{random.choice(files)}")


DEFAULT_SHAPE_DETECT_MODEL_PATH = "runs/detect/exp/weights/best.pt"

random_img = get_random_img()
img = cv.imread(random_img)

res = predict(DEFAULT_SHAPE_DETECT_MODEL_PATH, random_img)
tensor = res[0].boxes.xywh.clone().detach()
xywh_tensor_values = tensor.cpu().numpy().tolist()[0] 

cropped_image = crop_image(img, xywh_tensor_values) 
isolated_character_img = isolate_character(img, xywh_tensor_values) 

cv.imshow("isolated character", cv.imread("cluster_results.jpg"))
cv.waitKey(0)
cv.destroyAllWindows()
