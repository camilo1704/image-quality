# image-quality

## Project Overview

### Description
This project collects several image processing tecniques to analyze diverse properties that are often related with the quality of an image.

### Objective
Tag images with **blur**, **dark**, **long**, **black** and **bright** labels.

### Technologies Used
Image Processing libraries: Open-cv, PIL, Scipy, skimage.
ML libraries: sklearn, Pytorch.

## Demo

## Instalation 
### Clone the repository
git clone https://github.com/camilo1704/image-quality.git

### Install dependencies
pip install -r requirements.txt

## Project Structure
├── src/                                # Main source code
    ├── algorithms/                     # Algorithms for tagging images
        ├── algorithmic_clustering.py   
        └── blur.py                     
    ├── data_processing/                # Helping functions 
        └── compute_blur.py             
    ├── models/                         # Model training scripts and feature extractions methods
        ├── blur_model.py               
        ├── feature_extractors.py       
        └── image_attributes.py         
    ├── utils/                          # Helping functions
        ├── files.py                    
        ├── graphs.py                  
        └── models.py                   
├── main.py                             # Main Script for tagging images
├── requirements.txt                    # Requirements file
└── README.md                           # Project README

## Dataset
To train the blur model, 2 folders with blur and normal images need to be provided. 

## Usage
### Blur Classificator
python ./src/models/blur_model.py --images_path --save_artifacts_path

### image tagging
python ./main.py --images_path --output_path --blur_artifacts_path
