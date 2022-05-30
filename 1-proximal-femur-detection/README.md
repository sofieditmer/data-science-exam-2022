# Detection of the Proximal Femur with YOLOv5

## Description

This directory contains scripts and outputs related to the training and evaluation of YOLOv5 in detecting proximal femurs on X-rays of patients suffering from Legg-Calve-Perthes Disease. The train, validation and test data is not contained in this repository due to ethical and legal concerns.

## Directory Structure
```
|-- yolov5/                # YOLOv5 repository cloned from GitHub: https://github.com/ultralytics/yolov5
    |-- models/            # YOLOv5 model anchors
    |-- runs/              # Training and validation results (excluding those not sharable due to ethical and legal concerns)
    |-- utils/             # YOLOv5 utilities
    |-- train.py           # YOLOv5 training script, outputs in runs/train
    |-- val.py             # YOLOv5 validation script, outputs in runs/val
    |-- detect.py          # YOLOv5 detection script, used for detection and cropping, outputs not in this repository
    |-- requirements.txt   # YOLOv5 requirements, necessary to run YOLOv5 scripts
    |-- ...
|-- data-train.yaml        # .yaml file used for defining train and validation data for train.py script
|-- data-val.yaml          # .yaml file used for defining train, validation and test data for val.py script
|-- prepare_data.ipynb     # Notebook for preparation of data
|-- README.md              # README file providing information of the project
```

## Usage

Even though the data could not be provided, the following steps describe how training, evaluation and detection was completed.

**1 | Setting up environment**

To install relevant packages from the command line, the `requirements.txt` file, provided by YOLOv5 was used:

```
cd yolov5/
pip install -r requirements.txt
```

Since errors with tensorflow occured when running the scripts, the following the following packages and their associated versions were installed separately, by running the following commands:

```
pip install torch==1.8.1
pip install torchvision==0.9.1
```

**2 | Training of YOLOv5**

To train YOLOv5 on the X-ray images and their associated labels (bounding box coordinates, saved in .txt files), the following command was run from the command line:

```
python3 train.py --img 640 --batch 128 --epochs 300 --data ../data-train.yaml --weights yolov5l6.pt --name train
```

The output of this script is saved in `runs/train/train`.


**3 | Evaluation of YOLOv5**

To evaluate YOLOv5 on the validation and test data, the following commands were run from the command line: 

```
python3 val.py --img 640 --batch 128 --data ../data-val.yaml --weights runs/train/train/weights/best.pt --task val --name val
python3 val.py --img 640 --batch 128 --data ../data-val.yaml --weights runs/train/train/weights/best.pt --task test --name test
```

The outputs of this script is saved in `runs/val/val` and `runs/val/test`. 

**4 | Detection and Cropping of Proximal Femurs using the Trained YOLOv5** 

To detect and save cropped images of the proximal femurs, the following commands were run from the command line:

```
python3 detect.py --weights runs/train/train/weights/best.pt --img 640 --conf 0.8 --source ../../data/images/test --name detect --save-crop
```

Note that the above command only detects proximal femurs on the test data, but it was also run on the train and validation data. 


    

    
