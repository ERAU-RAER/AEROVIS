from dotenv import load_dotenv
import os
from ultralytics import YOLO
from roboflow import Roboflow
import argparse
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


def train(use_model, save_to, num_epochs, exp_name):
    os.makedirs('./models', exist_ok=True)
    model_path = os.path.abspath(use_model)
    model = YOLO(model_path)
    data_path = os.path.abspath("./standard_object_shape-1/data.yaml")
    model.train(model=model_path, data=data_path, save=True, epochs=num_epochs, name=exp_name) 
    full_save_path = os.path.abspath(save_to)
    os.replace("yolov8n.pt", full_save_path) 


def predict(use_model, imgs_to_predict):
    use_model_path = os.path.abspath(use_model) 
    imgs_dir = os.path.abspath(imgs_to_predict)
    infer = YOLO(use_model_path)
    infer.predict(source=imgs_dir, save=True, save_txt=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-T", "--train", action="store_true", help="Train a model.")
    parser.add_argument("-t", "--train-using", type=str, help="Train the specified model.")
    parser.add_argument("-a", "--save-as", type=str, help="Save model to specific path.")
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs.")
    parser.add_argument("-P", "--predict-images", action="store_true", help="Predict using a directory of images.")
    parser.add_argument("-i", "--images", type=str, help="Directory of images used to predict.")
    parser.add_argument("-u", "--use-model", type=str, help="Use the specified weights (path/to/best.pt).")
    parser.add_argument("-n", "--experiment-name", type=str, help="Name of experiment.")
    args = parser.parse_args()

    if args.train:
        if not os.path.exists("./standard_object_shape-1"):
            download_data()
        save_to = "./models/yolov8n.pt" if args.save_as is None else args.save_as
        num_epochs = 1 if args.epochs is None else args.epochs 
        train_using = "./models/yolov8n.pt" if args.train_using is None else args.train_using
        exp_name = "exp" if args.experiment_name is None else args.experiment_name
        train(train_using, save_to, num_epochs, exp_name)
    elif args.predict_images:
        use_model = "runs/exp/weights/best.pt" if args.use_model is None else args.use_model 
        img_dir = "standard_object_shape-1/test/images" if args.images is None else args.images
        predict(use_model, img_dir) 


if __name__ == "__main__":
    main()
