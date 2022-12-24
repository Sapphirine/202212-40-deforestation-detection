import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

output_directory = "normalized_dataset"
satellite_directory = "satellite_images"
mask_directory = "deforestation"
scenes = ["227063", "227065"]
years = ["2017", "2018"]

def run():
    for i in range(0,len(scenes)):
        satellite_url = ""
        for folder in os.listdir(satellite_directory):
            scene = folder.split("_")[2]
            year = folder.split("_")[3][:4]
            if scene==scenes[i] and year==years[0]:
                for file in os.listdir(os.path.join(satellite_directory, folder)):
                    if file.split("_")[-1] == "B2.TIF":
                        satellite_url = os.path.join(satellite_directory, folder, file.replace("B2.TIF", "B"))
                        break
                break
        create_patches_for_satellite(satellite_url, year, scene)

    for i in range(0,len(scenes)):
        for j in range(1,len(years)):
            satellite_url = ""
            mask_url = ""
            mask_year = ""
            for folder in os.listdir(satellite_directory):
                scene = folder.split("_")[2]
                year = folder.split("_")[3][:4]
                if scene==scenes[i] and year==years[j]:
                    for file in os.listdir(os.path.join(satellite_directory, folder)):
                        if file.split("_")[-1] == "B2.TIF":
                            satellite_url = os.path.join(satellite_directory, folder, file.replace("B2.TIF", "B"))
                            mask_url = os.path.join(mask_directory, f"{scene}_{year}.TIF")
                            mask_year = str(int(year[2:])-1) + "_" + year[2:]
                            break
                    break
            create_patches_for_scene(satellite_url, mask_url, scene, year, mask_year)

#for both satellite image and mask
def create_patches_for_scene(satellite_url, mask_url, scene, year, mask_year):
    bands = []
    for i in range (2, 8):
        band_url = satellite_url + str(i) + ".TIF"
        band = plt.imread(band_url)
        band = band / np.amax(band)
        # band = (band * 255).astype(np.uint8)
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

    if not os.path.exists(os.path.join(output_directory, year, scene)):
        os.makedirs(os.path.join(output_directory, year, scene))
        for i in range(len(valid_patches)):
            np.save(os.path.join(output_directory, year, scene, str(i)), tf.reshape(valid_patches[i], shape=(256,256,6)).numpy())

    if not os.path.exists(os.path.join(output_directory, mask_year, scene)):
        os.makedirs(os.path.join(output_directory, mask_year, scene))
        for i in range(len(valid_mask_patches)):
            np.save(os.path.join(output_directory, mask_year, scene, str(i)), tf.reshape(valid_mask_patches[i], shape=(256,256)).numpy())

#for just satellite image
def create_patches_for_satellite(satellite_url, year, scene):
    bands = []
    for i in range (2, 8):
        band_url = satellite_url + str(i) + ".TIF"
        band = plt.imread(band_url)
        band = band / np.amax(band)
        # band = (band * 255).astype(np.uint8)
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

    if not os.path.exists(os.path.join(output_directory, year, scene)):
        os.makedirs(os.path.join(output_directory, year, scene))
        for i in range(len(valid_patches)):
            np.save(os.path.join(output_directory, year, scene, str(i)), tf.reshape(valid_patches[i], shape=(256,256,6)).numpy())

if __name__=='__main__':
    run()
