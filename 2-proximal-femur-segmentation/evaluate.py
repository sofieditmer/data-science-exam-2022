"""
Script for evaluating best U-Net model on train, validation or test data. 
"""

# Importing Packages
import os, glob
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import keras
from keras.optimizers import Adam
from model import UNET
from utils import dice_score, dice_loss, plot_predictions


# -- MAIN FUNCTION ---

def main():
    
    # ARGUMENT PARSER 
    
    # Define arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--task", type = str, help = "Task, referring to val or test dataset",
                    required = True)
    ap.add_argument("-d", "--directory", type = str, help = "Directory containing model, and where results will be saved",
                    required = True)
    
    # Parse arguments
    args = vars(ap.parse_args()) 
    dataset = args["task"]
    directory = args["directory"]
    
    
    # PREPARE IMAGES AND MASKS
    
    # Load and prepare images
    images = []
    for image in sorted(glob.glob(f"../data/segmentation/{dataset}/images/*.jpg")):
        img = cv2.imread(image, 0)
        img = cv2.resize(img, (256,256))
        img = img/255
        img = np.reshape(img,img.shape + (1,))
        images.append(img)
    
    # Load and prepare masks
    masks = []
    for image in sorted(glob.glob(f"../data/segmentation/{dataset}/masks/*.jpg")):
        img = cv2.imread(image, 0)
        img = cv2.resize(img, (256,256))
        img = img/255
        img = np.reshape(img,img.shape + (1,))
        masks.append(img)
        
    # Save images and array in arrays
    images = np.array(images)
    masks = np.array(masks)
    
    
    # LOAD MODEL
    
    # Load model
    model = keras.models.load_model(f"{directory}/model.h5",
                                    custom_objects={'dice_loss': dice_loss, 'dice_score':dice_score})
    
    # Load best model weights
    model.load_weights(f"{directory}/checkpoints.h5")
    
    
    # EVALUATE MODEL
    
    # Evaluate the model on images and masks
    results = model.evaluate(images, masks)
    
    
    # SAVE RESULTS
    
    # Save metrics in txt file
    with open(f"{directory}/{dataset}_results.txt", "w") as f:
        f.writelines([f"Dice Loss: {results[0]}\n", 
                      f"Accuracy: {results[1]}\n",
                      f"Binary_Accuracy: {results[2]}\n",
                      f"Dice Score: {results[3]}\n",
                      f"IoU: {results[4]}\n",
                      f"Binary IoU: {results[5]}\n"])
    
    
    # GENERATE PREDICTIONS
    
    # Generate predictions from masks
    predictions = model.predict(images)
    # Threshold probabilities at 0.5
    predicted_masks = predictions > 0.5
    # Visualise predicted masks
    for index, img in enumerate(images):
        vis = plot_predictions(images[index], masks[index], predicted_masks[index])
        vis.savefig(f"{directory}/{dataset}_predictions/{index}.png")

        
if __name__=="__main__":
    main()
        
        

