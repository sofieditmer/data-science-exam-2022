# Segmentation of the Proximal Femur with U-Net

## Description

This directory contains scripts and outputs related to the training and evaluation of U-Net in segmenting proximal femurs on cropped X-rays (output of `1-proximal-femur-detection`) of Legg-Calve-Perthes Disease. The train, validation and test data is not contained in this repository due to ethical and legal concerns.

## Directory Structure
```
|-- notebooks/                          # Notebooks for prepocessing of data
    |-- convert-json-to-masks.ipynb     # Notebook for converting .json annotations to image masks
    |-- splitting_data.ipynb            # Notebook for splitting data into train, val and test
    |-- subsetting_data.ipynb           # Notebook for subsetting 200 cropped images from YOLOv5 output
|-- output/                             # Outputs of train.py and evaluate.py scripts
    |-- history.csv                     # Training history of model
    |-- history.png                     # History of accuracy over epochs
    |-- test_results.txt                # Evaluation results on test data, obtained from evaluate.py
    |-- val_results.txt                 # Evaluation results on val data, obtained from evaluate.py
|-- evaluate.py                         # Evaluation script to evaluate best trained model
|-- model.py                            # U-Net model architecture
|-- train.py                            # Script for training U-Net model
|-- utils.py                            # Utility functions for preparing data, training and visualisations
|-- README.md                           # README file for directory
```

## Usage

Even though the data could not be provided, the following steps describe how training and evaluation was completed.

**1 | Setting up environment**

To install relevant packages from the command line, the `requirements.txt` file, provided in the root directory was used:

```
pip install -r requirements.txt
```

**2 | Training of U-Net**

To train U-NET on the cropped X-ray images and their associated labels (segmentation masks, saved as images), the following command was run from the command line:

```
python3 train.py --img_size 256 --img_channels 1 --learning_rate 0.00001 --epochs 400 --batch_size 32
```

The output of this script is saved in `output/`.


**3 | Evaluation of U-Net**

To evaluate the best U-Net on the validation and test data, the following commands were run from the command line: 

```
python3 evaluate.py --task val --directory output
python3 evaluate.py --task test --directory output
```

The outputs of this script is saved in `output`. The metrics for the training data on the best model, which are reported in the paper, are extracted in `assets/paper_visualisations.ipynb`. Note that the visualisations of the masks could not be provided due to legal and ethical concerns. 

    

    

