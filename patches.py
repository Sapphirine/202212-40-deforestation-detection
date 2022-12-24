import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

output_directory = ""
satellite_directory = ""
mask_directory = ""
scenes = []
years = []

def run():
    pass

#for both satellite image and mask
def create_patches_for_scene(satellite_url, mask_url, scene, year, mask_year):
    bands = []
    for i in range (2, 8):
        band_url = satellite_url + str(i) + ".tif"
        band = plt.imread(band_url)
        band = band / np.amax(band)
        band = (band * 255).astype(np.uint8)
        bands.append(band)
    img = np.stack(bands, axis=-1)

    mask = plt.imread(mask_url)
    mask = mask[:,:,0].reshape(img.shape[0],img.shape[1],1)

    patches = tf.image.extract_patches([img], [1,256,256,1], [1,256,256,1], [1,1,1,1], padding='VALID')
    mask_patches = tf.image.extract_patches([mask], [1,256,256,1], [1,256,256,1], [1,1,1,1], padding='VALID')
    valid_patches = []
    valid_mask_patches = []
    for imgs in patches:
        for r in range(patches.shape[1]):
            for c in range(patches.shape[2]):
    #             black_pixels = len(tf.where(imgs[r,c] == 0))
    #             black_pixels_ratio = float(black_pixels) / patches.shape[3]
    #             if black_pixels_ratio < .1:
                valid_patches.append(imgs[r,c])
                valid_mask_patches.append(mask_patches[0,r,c])

    for i in range(len(valid_patches)):
        np.save(os.path.join(output_directory, year, scene, str(i)), tf.reshape(valid_patches[i], shape=(256,256,6)).numpy())

    for i in range(len(valid_mask_patches)):
        np.save(os.path.join(output_directory, mask_year, scene, str(i)), tf.reshape(valid_mask_patches[i], shape=(256,256)).numpy())

#for just satellite image
def create_patches_for_satellite(satellite_url, year, scene):
    bands = []
    for i in range (2, 8):
        band_url = satellite_url + str(i) + ".tif"
        band = plt.imread(band_url)
        band = band / np.amax(band)
        band = (band * 255).astype(np.uint8)
        bands.append(band)
    img = np.stack(bands, axis=-1)

    patches = tf.image.extract_patches([img], [1,256,256,1], [1,256,256,1], [1,1,1,1], padding='VALID')
    valid_patches = []
    for imgs in patches:
        for r in range(patches.shape[1]):
            for c in range(patches.shape[2]):
    #             black_pixels = len(tf.where(imgs[r,c] == 0))
    #             black_pixels_ratio = float(black_pixels) / patches.shape[3]
    #             if black_pixels_ratio < .1:
                valid_patches.append(imgs[r,c])

    for i in range(len(valid_patches)):
        np.save(os.path.join(output_directory, year, scene, str(i)), tf.reshape(valid_patches[i], shape=(256,256,6)).numpy())