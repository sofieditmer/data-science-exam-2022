"""
Script for training and saving U-NET Model
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
from keras.metrics import IoU, BinaryIoU
from model import UNET
from utils import prepare_data, image_mask_generator, dice_score, dice_loss, plot_history


# --- MAIN FUNCTION ---

def main():
    
    # ARGUMENT PARSER
    
    # Define arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--name", type = str, help = "Name of model, will create folder with name",
                    required = False, default = "output")
    
    ap.add_argument("-s", "--img_size", type = int, help = "H and W of image",
                    required = False, default = 256)
    
    ap.add_argument("-c", "--img_channels", type = int, help = "Number of image channels",
                    required = False,default = 1)
    
    ap.add_argument("-l", "--learning_rate", type = float, help = "Learning rate for model",
                    required = False,default = 0.00001)
    
    ap.add_argument("-e", "--epochs", type = int, help = "Number of epochs for training",
                    required = False, default = 100)
    
    ap.add_argument("-b", "--batch_size", type = int, help = "Batch size for training and validating",
                    required = False, default = 32)
    
    # Parse arguments
    args = vars(ap.parse_args())    
    name = args["name"]
    img_size = args["img_size"]
    img_channels = args["img_channels"]
    learning_rate = args["learning_rate"]
    epochs = args["epochs"]
    batch_size = args["batch_size"]
    
    
    # PREPARE OUTPUT DIRECTORY
    
    if not os.path.exists(os.path.join(name)):
        os.mkdir(os.path.join(name))
    
    
    # TRAIN DATA AND AUGMENTATION
    
    # Dictionary for data generators
    data_gen_args = dict(rotation_range=10,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    
    # Training generator
    train_path = "../data/segmentation/train/"
    train_gen = image_mask_generator(data_gen_args, train_path, 
                                     target_size=(img_size,img_size), 
                                     batch_size=batch_size)
    
    
    # VALIDATION DATA
    
    # Load and prepare images
    val_images = []
    for image in sorted(glob.glob("../data/segmentation/val/images/*.jpg")):
        img = cv2.imread(image, 0)
        img = cv2.resize(img, (256,256))
        img = img/255
        img = np.reshape(img,img.shape + (1,))
        val_images.append(img)

    # Load and prepare masks
    val_masks = []
    for image in sorted(glob.glob("../data/segmentation/val/masks/*.jpg")):
        img = cv2.imread(image, 0)
        img = cv2.resize(img, (256,256))
        img = img/255
        img[img > 0.5] = 1
        img[img <= 0.5] = 0
        img = np.reshape(img,img.shape + (1,))
        val_masks.append(img)
        
    # Save in arrays
    val_images = np.array(val_images)
    val_masks = np.array(val_masks)
    
    # Print shape of array
    print(val_masks.shape)
    
    
    # DEFINE AND PREPARE MODEL

    # Load model
    model = UNET(None, (img_size, img_size, img_channels))
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss=dice_loss, metrics=['accuracy', 'binary_accuracy', dice_score, 
                                           IoU(num_classes=2, target_class_ids=[1]),
                                           BinaryIoU(target_class_ids=[1], threshold=0.5)])
    
    # Save checkpoints of best model
    callbacks = [keras.callbacks.ModelCheckpoint(f"{name}/checkpoints.h5", save_best_only=True,
                                                 monitor='val_loss', mode='min')]
    
    # Train model
    history = model.fit_generator(generator=train_gen,  
                                  steps_per_epoch=80/batch_size,
                                  validation_data=(val_images, val_masks),
                                  epochs=epochs,
                                  verbose=1, 
                                  shuffle=True,
                                  callbacks=callbacks)
    
    
    # SAVE RESULTS
    
    # Save model
    model.save(f"{name}/model.h5")
    
    # Plot the history
    plot_history(history, epochs, f"{name}/history.png")
    
    # Save the history to data frame
    hist_df = pd.DataFrame(history.history)
    hist_csv_file = f"{name}/history.csv"
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    

if __name__=="__main__":
    main()