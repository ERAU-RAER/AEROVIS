# AEROVIS

AErial Reconnaissance and Evaluation Operation VIsion System

This project is a subcomponent of [AEROLOGIC](https://github.com/ERAU-SUAS/AEROLOGIC).

## Shape Detector  

### Getting Started

1. Create an account on https://universe.roboflow.com.

2. Create a workspace. [docs](https://docs.roboflow.com/workspaces/roboflow-workspaces)

3. Create an _object detection_ project. [docs](https://docs.roboflow.com/datasets/create-a-project)

4. Generate a private API key and copy it. [docs](https://docs.roboflow.com/api-reference/authentication) 

5. Create a file called `.env`. 

6. Add the line `ROBOFLOW_API_KEY=<your private key>`.

7. Install dependencies using `pip install -r requirements.txt`. 

### Usage 

#### Train

- **-T, --train** 
Train a new model.

- **-t, --train-using [path/to/model.pt]** 
Train using specified model. Default: `models/yolov8n.pt`.

- **-s, --save-as [desired/path]**
Save the model to a specified path. Default: `models/yolov8n.pt`.

- **-e, --epochs [num_epochs]** 
Define number of training epochs. Default: 1.

- **-n, --experiment-name [name]**
Name of experiment directory. Default: `exp`. 

#### Predict

- **-P, --predict-images**
Predict a directory of images using a defined model. Default is `

- **-i, --images [path/to/image/dir]**
Image directory to predict. Default: `standard_object_shape-1/test/images`.

- **-u, --use-model [path/to/model/weights.pt]**
Model to use for predictions. Default: `runs/exp/

### Usage

#### Train 

| Option | Description | Default |
|---|---|---|
| `-T`, `--train` | Train a new model. | N/A |
| `-t`, `--train-using` | Train using specified model. | `models/yolov8n.pt` |
| `-s`, `--save-as` | Save the model to a specified path. | `models/yolov8n.pt` |
| `-e`, `--epochs` | Define number of training epochs. | 1 |
| `-n`, `--experiment-name` | Name of experiment directory. | `exp` |

#### Predict 

| Option | Description | Default |
|---|---|---|
| `-P`, `--predict-images` | Predict a directory of images using a defined model. | N/A |
| `-i`, `--images` | Image directory to predict. | `standard_object_shape-1/test/images` |
| `-u`, `--use-model`| Model to use for predictions. | `runs/exp/weights/best.pt` |
