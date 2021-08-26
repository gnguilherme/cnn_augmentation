import os
import glob
import json

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import BatchNormalization, Dropout, MaxPooling2D
from tensorflow.keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


AUTOTUNE = tf.data.AUTOTUNE
with open('src/config.json', 'rb') as file:
    config = json.load(file)
EPOCHS = config['epochs']
MAIN_PATH = config['main_path']
BATCH_SIZE = config['batch_size']
MODEL_PATH = config['model_path']
INPUT_SHAPE = config['input_shape']
CONV_FILTERS = config['conv_filters']
DENSE_FILTERS = config['dense_filters']
LEARNING_RATE = config['learning_rate']


def load_image(path: str):
    """
    Load image and label from file

    Parameters
    ----------
    path: str
        file path

    Returns
    -------
    tensorflow Tensor
        image loaded
    tensorflow Tensor
        target label
    """

    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=INPUT_SHAPE[-1])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, INPUT_SHAPE[:2])
    label = tf.strings.split(path, os.path.sep)[-2]
    label = tf.strings.to_number(label, tf.int32)

    return image, label


def augment_data(images, labels):
    """
    Augment data

    Parameters
    ----------
    images: tensorflow Tensor
        target images
    labels: tensorflow Tensor
        target labels

    Returns
    -------
    tensorflow Tensor
        image loaded
    tensorflow Tensor
        target label
    """

    images = tf.image.random_flip_up_down(images, seed=42)
    images = tf.image.random_flip_left_right(images, seed=42)
    images = tf.image.random_hue(images, 0.05, seed=42)
    images = tf.image.random_brightness(images, 0.05, seed=42)

    return images, labels


def get_dataset(step: str = 'train', augment: bool = False):
    """
    Load dataset

    Parameters
    ----------
    step: str
        step. Default: 'train'
    augment: bool
        use data augmentation? Default: False

    Returns
    -------
    tensorflow Dataset
        step dataset
    """

    paths = glob.glob(MAIN_PATH + f'/{step}/*/*')

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.shuffle(len(paths), seed=42) \
        .map(load_image, num_parallel_calls=AUTOTUNE) \
        .cache() \
        .batch(BATCH_SIZE)

    if augment:
        ds = ds.map(augment_data, num_parallel_calls=AUTOTUNE)

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def create_model():
    """
    Create CNN model

    Returns
    -------
    tensorflow Model
        compiled model
    """

    tin = Input(INPUT_SHAPE)

    for i, f in enumerate(CONV_FILTERS):
        conv_layer = Conv2D(filters=f, kernel_size=(3, 3), padding='same',
                            kernel_initializer='he_normal')
        if not i:
            conv_layer = conv_layer(tin)
        else:
            conv_layer = conv_layer(dropout)
        activation = ReLU()(conv_layer)
        batch_norm = BatchNormalization()(activation)
        max_pool = MaxPooling2D(pool_size=(2, 2))(batch_norm)
        dropout = Dropout(rate=0.2)(max_pool)

    flattened = GlobalAveragePooling2D()(dropout)
    for i, f in enumerate(DENSE_FILTERS):
        dense = Dense(units=f, kernel_initializer='he_normal')
        if not i:
            dense = dense(flattened)
        else:
            dense = dense(dropout)
        activation = ReLU()(dense)
        batch_norm = BatchNormalization()(activation)
        dropout = Dropout(rate=0.3)(batch_norm)

    output = Dense(units=1, activation='sigmoid')(dropout)

    model = Model(inputs=[tin], outputs=[output])

    model.compile(
        optimizer=Adam(lr=LEARNING_RATE, decay=LEARNING_RATE / EPOCHS),
        metrics=[Recall(), Precision(), 'accuracy'],
        loss=BinaryCrossentropy(),
    )
    model.summary()

    return model


def train_model(model, datasets: list):
    """
    Train CNN model

    Parameters
    ----------
    model: tensorflow Model
        compiled model
    datasets: list
        list of datasets
    """

    train, validation, test = datasets
    model_path = os.path.join(MODEL_PATH, 'model')
    mc = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        filepath=model_path,
        verbose=2,
        save_weights_only=True,
        save_best_only=True,
    )
    es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10,
        verbose=2,
        restore_best_weights=True,
    )
    try:
        model.load_weights(model_path)
        print("Model weights loaded")
    except Exception as e:
        print(e)
        print("Weights not loaded")
    history = model.fit(
        train,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        validation_data=validation,
        class_weight={0: 0.5, 1: 2},
        callbacks=[mc, es],
    )
    print(f"Model evaluation")
    model.evaluate(test)

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    axes[0].plot(loss, label='Loss')
    axes[0].plot(val_loss, label='Validation Loss')
    axes[0].legend(frameon=False)
    axes[1].plot(accuracy, label='Accuracy')
    axes[1].plot(val_accuracy, label='Validation Accuracy')
    axes[1].legend(frameon=False)
    plt.suptitle(f"Train/Validation loss/Accuracy")

    plt.tight_layout()
    plt.show()


def main():

    train_ds = get_dataset(step='train', augment=True)
    validation_ds = get_dataset(step='validation')
    test_ds = get_dataset(step='test')

    datasets = [train_ds, validation_ds, test_ds]

    model = create_model()

    train_model(model, datasets)


if __name__ == '__main__':
    main()
