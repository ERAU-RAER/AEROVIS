import os
import random
from src.shape_detector import predict

def get_random_img():
    dir = os.path.abspath("standard_object_shape-1/test/images") 
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    return os.path.abspath(f"standard_object_shape-1/test/images/{random.choice(files)}")


DEFAULT_SHAPE_DETECT_MODEL_PATH = "runs/detect/exp/weights/best.pt"

res = predict(DEFAULT_SHAPE_DETECT_MODEL_PATH, get_random_img())

tensor = res[0].boxes.xywh.clone().detach()
tensor_values = tensor.cpu().numpy().tolist()

print(tensor_values)

