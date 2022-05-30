# Computer Vision for Medical Image Analysis: Detection and Segmentation of  the Proximal Femur in Legg-Calvé-Perthes Disease using Deep Learning

## Project Description
This repository contains the contents of an exam project in the course Data Science at the Master's degree in Cognitive Science at Aarhus University. 
More specifically it contains the relevant scripts for: 

1. **Detection and localisation** of the proximal femur in X-ray images of Legg-Calvé-Perthes Disease patients using YOLOv5
2. **Semantic segmentation** of the proximal femur on cropped images of detected proximal femurs obtained in (1)


## Repository Structure
```
|-- 1-proximal-femur-detection/       # Directory containing the main scripts for object detection using YOLOv5
                                      # See README.md in directory for detailed information

|-- 2-proximal-femur-segmentation/    # Directory containing the main scripts for semantic segmentation using U-Net
                                      # See README.md in directory for detailed information
                                      
|-- assets/                           # Directory containing notebook and output for visualisations
|-- install-requirements.sh           # Bash script for installing necessary dependencies
|-- requirements.txt                  # Necessary dependencies to run scripts and notebooks
|-- README.md                         # Main README file for repository

```

## Usage
**!** The scripts have only been tested on Linux, using Python 3.9.10.
Due to ethical and legal concerns, the data of this project cannot be shared on this repository. Nevertheless, scripts and directions on how the scripts were used are provided in the README.md files of the subdirectories. 

## Contact
This project was developed by Louise Nyholm Jensen, Nicole Dwenger and Sofie Ditmer. 