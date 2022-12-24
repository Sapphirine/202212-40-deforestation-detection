import tensorflow as tf
import numpy as np
import os
from resunet import ResUNet
from keras.callbacks import CSVLogger
from keras.models import load_model

input_directory = "normalized_dataset"
train_scenes = ["227063", "227065", "225063", "003066", "233067"]
years = ["17_18", "18_19"]
num_patches = 870
epochs = 350
val_size = .15
metrics = [tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
loss = "categorical_crossentropy"
batch = 16

def train(model_path, log_path):
    dataset = tf.data.Dataset.from_generator(
        get_train_patches,
        output_types=(tf.float32, tf.float32),
        output_shapes=([256,256,12], [256,256])
    ).shuffle(2000)
    train_size = int(num_patches * len(train_scenes) * len(years) * (1 - val_size))
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    csv_logger = CSVLogger(f'{log_path}.csv', append=True, separator=',')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_path, monitor='loss', save_best_only=True
    )

    model = ResUNet(input_shape=(256,256,12), classes=1, filters_root=16, depth=3)
    model.compile(loss=loss, optimizer="adam", metrics=metrics)
    model.fit(
        train_dataset.batch(batch),
        validation_data=val_dataset.batch(batch),
        epochs=epochs,
        callbacks=[csv_logger, checkpoint])

    model.save(model_path)

def train_from_checkpoint(load_model_path, save_model_path, log_path):
    dataset = tf.data.Dataset.from_generator(
        get_train_patches,
        output_types=(tf.float32, tf.float32),
        output_shapes=([256,256,12], [256,256])
    ).shuffle(2000)
    train_size = int(num_patches * len(train_scenes) * len(years) * (1 - val_size))
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    csv_logger = CSVLogger(f'{log_path}.csv', append=True, separator=',')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        save_model_path, monitor='loss', save_best_only=True
    )

    model = load_model(load_model_path)
    model.fit(
        train_dataset.batch(batch),
        validation_data=val_dataset.batch(batch),
        epochs=epochs,
        callbacks=[csv_logger, checkpoint])

    model.save(save_model_path)

def get_train_patches():
    for year in years:
        for scene in train_scenes:
            year1path = os.path.join(input_directory, "20"+year[:2], scene)
            year2path = os.path.join(input_directory, "20"+year[3:], scene)
            maskpath = os.path.join(input_directory, year, scene)
            for i in range(num_patches):
                img1 = np.load(os.path.join(year1path, f"{i}.npy"))
                img2 = np.load(os.path.join(year2path, f"{i}.npy"))
                mask = np.load(os.path.join(maskpath, f"{i}.npy"))
                yield np.concatenate([img1,img2], axis=2), mask

if __name__=='__main__':
    train(os.path.join("models", "moredata_norm_cross_350"),
        os.path.join("models", "moredata_norm_cross"))
    # train_from_checkpoint(os.path.join("models",  "moredata_norm_cross_350"),
    #     os.path.join("models",  "moredata_norm_cross_lots"),
    #     os.path.join("models",  "moredata_norm_cross"))