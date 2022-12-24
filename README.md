# 202212-40-deforestation-detection

## Setup

ensure the following packages are installed:
tensorflow
matplotlib
numpy
git+https://github.com/JanPalasek/resunet-tensorflow

## Downloading images
landsat.py will download the satellite scenes from scenes.txt, but that requires a USGS account with access to their machine-to-machine API, so I've included some sample images in the satellite_images folder

## Generating deforestation masks
This step does **not** work in PyQGIS. My attempt at it is in create_mask.ipynb. The outputs of this step were created manually using the QGIS GUI app, with some sample mask images in the deforestation folder.

## Creating patches
run `python patches.py`

## Training the model
run `python train.py`

## Testing the model
run `python test.py`
