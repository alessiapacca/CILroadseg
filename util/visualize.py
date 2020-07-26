
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

# x - data matrix of shape (w, h, 3) corresponding to a single RGB image
def view_image(x):
    plt.imshow(Image.fromarray(np.uint8(x * 255)))

# y - data matrix of shape (w, h) corresponding to a single grayscale image
def view_mask(y):
    plt.imshow(Image.fromarray(np.uint8(y * 255), 'L'))

# x - data matrix of shape (w, h, 3) corresponding to a single RGB image
# y - data matrix of shape (w, h) corresponding to a single grayscale image
def view_image_mask(x, y):
    plt.imshow(Image.fromarray(np.uint8(x * 255)))
    plt.figure()
    plt.imshow(Image.fromarray(np.uint8(y * 255), 'L'))

def view_image_mask2(x, y, y_true):
    plt.imshow(Image.fromarray(np.uint8(x * 255)))
    plt.figure()
    plt.imshow(Image.fromarray(np.uint8(y * 255), 'L'))
    plt.figure()
    plt.imshow(Image.fromarray(np.uint8(y_true * 255), 'L'))


def view_image_array(X, Y1, Y2=None, Y3=None):
    cols = 2
    if Y2 is not None:
        cols += 1

    if Y3 is not None:
        cols += 1

    fig = plt.figure(figsize=(30, X.shape[0] * 10))
    for i in range(X.shape[0]):
        fig.add_subplot(X.shape[0], cols, cols * i + 1)
        plt.imshow(X[i])

        fig.add_subplot(X.shape[0], cols, cols * i + 2)
        plt.imshow(Y1[i])

        next_col = 3
        if Y2 is not None:
            fig.add_subplot(X.shape[0], cols, cols * i + next_col)
            plt.imshow(Y2[i])
            next_col += 1

        if Y3 is not None:
            fig.add_subplot(X.shape[0], cols, cols * i + next_col)
            plt.imshow(Y3[i])

    plt.show()


def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()