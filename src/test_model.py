import os
import json

import tensorflow as tf
import matplotlib.pyplot as plt

from src.train import create_model, get_dataset

AUTOTUNE = tf.data.AUTOTUNE
with open('src/config.json', 'rb') as file:
    config = json.load(file)

MODEL_PATH = config['model_path']


def main():
    test_ds = get_dataset(step='test')

    model = create_model()
    model_path = os.path.join(MODEL_PATH, 'model')
    try:
        model.load_weights(model_path)
        print("Model weights loaded")
    except Exception as e:
        print(e)
        print("Weights not loaded")
    model.evaluate(test_ds)

    for batch in test_ds.as_numpy_iterator():
        for img, label in zip(batch[0], batch[1]):
            plt.imshow(img, interpolation='bicubic')
            plt.suptitle(f"Prediction: {label}")
            plt.tight_layout()
            plt.show()
        break


if __name__ == '__main__':
    main()
