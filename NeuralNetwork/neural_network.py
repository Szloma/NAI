"""
Dawid Litwiński, Łukasz Kapkowski

neural network classification for wine, animals CIFAR10, mnist-fashion
 and mnist (0-9) - mnist is compared with 2 different sizes, small network with accuracy ~87%, bigger ~95%

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
mnist = tf.keras.datasets.mnist

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')

# Wine
def wine():
    df = pd.read_csv('wine.csv', sep=';', quotechar='"')
    X = df.drop(columns='quality')
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=1,
        stratify=y
    )
    model_wine = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(X.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(len(np.unique(y)), activation='softmax')
    ])

    model_wine.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model_wine.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=0)
    loss, acc = model_wine.evaluate(X_test, y_test, verbose=0)
    print(f"Wine accuracy: {acc:.3f}")



def animals():
    print("CIFAR-10 - animals")
    class_names = [
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse"
    ]
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    animal_classes = [0, 1, 2, 3, 4, 5]

    train_mask = np.isin(y_train.flatten(), animal_classes)
    test_mask = np.isin(y_test.flatten(), animal_classes)

    x_train, y_train = x_train[train_mask], y_train[train_mask]
    x_test, y_test = x_test[test_mask], y_test[test_mask]

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model_cifar = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(animal_classes), activation='softmax')
    ])

    model_cifar.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model_cifar.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

    img = x_test[0]
    label_true = y_test[0][0]

    img_batch = np.expand_dims(img, axis=0)

    predictions = model_cifar.predict(img_batch)
    predicted_class = np.argmax(predictions)

    print("Predykcja:", class_names[predicted_class])
    print("Prawdopodobieństwa:", predictions)
    y_pred = model_cifar.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)


    cm = confusion_matrix(y_test, y_pred_classes)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix – CIFAR-10 Animals")
    plt.show()

def fashion():
    print("Fashion MNIST")

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)

    model_fashion = keras.Sequential([
        layers.Flatten(input_shape=(28,28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model_fashion.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model_fashion.fit(x_train, y_train, epochs=10, validation_split=0.2)

def mnist():
    print("MNIST")

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    model = keras.Sequential([
        layers.Flatten(input_shape=(28,28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(x_train, y_train, epochs=5, validation_split=0.2)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n MINST 128-64-10 Test accuracy: {test_acc:.4f}")

def mnist2():
    print("MNIST")

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    model = keras.Sequential([
        layers.Flatten(input_shape=(28,28)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(x_train, y_train, epochs=5, validation_split=0.2)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n MINST 64-32-10 Test accuracy: {test_acc:.4f}")


def main():
   # wine()
   #  animals()
   #  fashion()
   # mnist2()
if __name__ == "__main__":
    main()