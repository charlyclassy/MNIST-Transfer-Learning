"""Train a purpose-built 28x28 handwritten digit CNN on MNIST.
Run: python train_model.py --epochs 15
"""
import argparse
from pathlib import Path
import json
import os
import random
import numpy as np
import tensorflow as tf


def build_model():
    inp = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.RandomRotation(0.06, fill_mode='constant')(inp)
    x = tf.keras.layers.RandomTranslation(0.08, 0.08, fill_mode='constant')(x)
    x = tf.keras.layers.RandomZoom(0.08, fill_mode='constant')(x)
    for filters in (32, 64):
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inp, out, name='DigitVisionCNN')
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--output', default='digitvision_cnn.keras')
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(42)
    random.seed(42)
    np.random.seed(42)
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32')[..., None] / 255.0
    rng = np.random.default_rng(42)
    ix = rng.permutation(len(x_train))
    validation_indices, training_indices = ix[:6000], ix[6000:]
    model = build_model()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=1e-5),
    ]
    history = model.fit(x_train[training_indices], y_train[training_indices],
                        validation_data=(x_train[validation_indices], y_train[validation_indices]),
                        epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks)
    model.save(str(output))
    print(f'Saved: {output.resolve()}')
    print(f'Best validation accuracy: {max(history.history["val_accuracy"]):.4%}')
    print('Run python evaluate_model.py to report untouched MNIST test accuracy.')


if __name__ == '__main__':
    main()
