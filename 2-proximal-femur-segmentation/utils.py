"""
Utility Functions for Training of U-NET Model
"""

# Import Packages
import os, glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator


# --- DATA PREPARATION ---

def prepare_data(img, mask):
    
    if np.max(img) > 1:
        # Normalisation
        img = img/255
        mask = mask/255
        # Binarising mask
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    
    return (img, mask)


# --- DATA AUGMENTATION ---

def image_mask_generator(aug_dict, data_path, target_size, batch_size):
    
    # Define generators
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    # Define generator for images
    image_generator = image_datagen.flow_from_directory(
        data_path,
        classes = ["images"],
        class_mode = None,
        color_mode = "grayscale",
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = "../data/segmentation/train/aug",
        save_prefix  = "image",
        shuffle= False,
        seed = 3)
    
    # Define generator for masks
    mask_generator = mask_datagen.flow_from_directory(
        data_path,
        classes = ["masks"],
        class_mode = None,
        color_mode = "grayscale",
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = "../data/segmentation/train/aug",
        save_prefix  = "mask",
        shuffle = False,
        seed = 3)
    
    # Zip images and masks
    train_generator = zip(image_generator, mask_generator)
    
    # Prepare images and masks
    for (img, mask) in train_generator:
        img, mask = prepare_data(img,mask)
        yield (img,mask)

        
# --- LOSS AND EVALUATION METRICS FOR SEGMENTATION ---

# Dice Coefficient
# Adapted from https://github.com/rekon/Smoke-semantic-segmentation/blob/linknet-implementation/LinkNet.ipynb
def dice_score(y_true, y_pred, smooth=1):
    intersection = keras.backend.sum(y_true * y_pred, axis=[1,2,3])
    union = keras.backend.sum(y_true, axis=[1,2,3]) + keras.backend.sum(y_pred, axis=[1,2,3])
    return keras.backend.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

# Dice Loss
def dice_loss(y_true, y_pred):
    return 1 - dice_score(y_true, y_pred)
  
    
# --- PROCESSING RESULTS ---        

# Plotting training history
def plot_history(H, epochs, output_path):
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)


# Plotting image and predictions 
# Adapted from https://github.com/sauravmishra1710/U-Net---Biomedical-Image-Segmentation/blob/main/UNet%20In%20Action.ipynb
def plot_predictions(image, mask, prediction_img = None):
    
    plt.close()
    
    fig = plt.figure(figsize=(20,20))
    fig.subplots_adjust(hspace = 0.6, wspace = 0.2)
    fig.subplots_adjust(top = 1.15)
    
    ax = fig.add_subplot(1, 4, 1)
    ax.imshow(image, cmap="gray")
    ax.title.set_text("Image")
    # remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax = fig.add_subplot(1, 4, 2)
    ax.imshow(np.reshape(mask, (256, 256)), cmap = "gray")
    ax.title.set_text("True Mask")
    # remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    if prediction_img is not None:
        ax = fig.add_subplot(1, 4, 3)
        ax.imshow(np.reshape(prediction_img, (256, 256)), cmap = "gray")
        ax.title.set_text("Predicted Mask")
        # remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax = fig.add_subplot(1, 4, 4)
        ax.imshow(image, cmap="gray")
        ax.imshow(np.reshape(prediction_img, (256,256)), cmap="gray", alpha=0.3)
        ax.title.set_text("Predicted Mask on Image")
        # remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    return fig