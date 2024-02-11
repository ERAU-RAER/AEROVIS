from dotenv import load_dotenv
import os
from ultralytics import YOLO
from roboflow import Roboflow
import sys 
import re 


def download_data():
    load_dotenv()
    roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")
    roboflow = Roboflow(api_key=roboflow_api_key)
    roboflow_project = roboflow.workspace("hku-uas").project("standard_object_shape")
    roboflow_project.version(1).download(model_format="yolov8")

    # change "train" and "valid" directories in data.yaml 
    path = os.path.abspath("./standard_object_shape-1/data.yaml")
    pattern = re.compile(r'standard_object_shape-1/') 
    with open(path, 'r') as file:
        new_lines = [pattern.sub("", line) for line in file]
    with open(path, 'w') as file:
        file.writelines(new_lines)


def train():
    model = YOLO("./models/yolov8n.pt")
    path = os.path.abspath("./standard_object_shape-1/data.yaml")
    model.train(data=path, epochs=3)


def classify():
    pass


def main():
    args = sys.argv
    for arg in args:
        if arg == "train":
            if not os.path.exists("./standard_object_shape-1"):
                download_data()
            train()
        if arg == "classify" and len(args) == 3:
            classify(args[2])

if __name__ == "__main__":
    main()
