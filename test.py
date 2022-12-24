from keras.models import load_model
import tensorflow as tf
import os
import numpy as np

input_directory = "normalized_dataset"
test_scenes = ["227065"]
years = ["17_18"]
num_patches = 870
batch = 16

def test_model(model_path):
    dataset = tf.data.Dataset.from_generator(
        get_test_patches,
        output_types=(tf.float32, tf.float32),
        output_shapes=([256,256,12], [256,256])
    )

    model = load_model(model_path)
    result = model.evaluate(dataset.batch(batch))
    print(result)

def get_test_patches():
    for year in years:
        for scene in test_scenes:
            year1path = os.path.join(input_directory, "20"+year[:2], scene)
            year2path = os.path.join(input_directory, "20"+year[3:], scene)
            maskpath = os.path.join(input_directory, year, scene)
            for i in range(num_patches):
                img1 = np.load(os.path.join(year1path, f"{i}.npy"))
                img2 = np.load(os.path.join(year2path, f"{i}.npy"))
                mask = np.load(os.path.join(maskpath, f"{i}.npy"))
                yield np.concatenate([img1,img2], axis=2), mask

if __name__ == '__main__':
    test_model(os.path.join("test_model"))
