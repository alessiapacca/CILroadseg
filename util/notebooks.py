
import sys
import os
import re

import numpy as np

from util.helpers import load_image
from util.visualize import save_image
from util.helpers import recompose


def nb_integrate_colab(dir):
    try:
        from google.colab import drive

        print('Colab environment detected. Mounting drive...')
        drive.mount('/content/drive')

        print('Mounted. Switching to directory... ', end = '')
        os.chdir('/content/drive/''My Drive''/'+ dir)
        print('done.')
    except:
        print('Colab environment not found. Working on ordinary directory.')

def nb_load_data(train_dir, train_gt_dir, test_dir):
    X = []
    Y = []
    X_test = []

    print("Loading training input...")
    files = sorted(os.listdir(train_dir))
    numfiles = len(files)
    k = 0
    for file in files:
        # print(file)
        X.append(np.asarray(load_image((train_dir + file))))
        k += 1

        sys.stdout.write("\rProgress: " + str(k * 100 // numfiles) + "%")

    print("\rProgress: done (" + str(len(X)) + " images).")
    X = np.array(X)

    print("Loading training groundtruth...")
    files = sorted(os.listdir(train_gt_dir))
    numfiles = len(files)
    k = 0
    for file in files:
        # print(file)
        Y.append(np.asarray(load_image((train_gt_dir + file))))
        k += 1

        sys.stdout.write("\rProgress: " + str(k * 100 // numfiles) + "%")

    print("\rProgress: done (" + str(len(Y)) + " images).")
    Y = (np.array(Y) >= 0.25) * 1  # compensates for lossy image data

    print("Loading test input...")
    files = sorted(os.listdir(test_dir))
    numfiles = len(files)
    k = 0
    for file in files:
        X_test.append(np.asarray(load_image((test_dir + file))))
        k += 1

        sys.stdout.write("\rProgress: " + str(k * 100 // numfiles) + "%")

    print("\rProgress: done (" + str(len(X_test)) + " images).")
    X_test = np.array(X_test)

    print()
    print("       Training data shape: " + str(X.shape))
    print("Training groundtruth shape: " + str(Y.shape))
    print("           Test data shape: " + str(X_test.shape))

    return X, Y, X_test

def nb_save_model(model, weights_file):
    print("[Target file: " + weights_file + "]")
    print("Saving model to disk...", end='')
    model.save(weights_file)
    print("done.")

def nb_predict_masks(model, test_dir, test_masks_dir):
    print("Predicting test cases... ")

    try:
        os.mkdir(test_masks_dir)
    except:
        pass

    files = os.listdir(test_dir)
    numfiles = len(files)

    k = 0
    for file in files:
        img_id = int(re.search(r"\d+", file).group(0))

        X_test = np.array([np.asarray(load_image((test_dir + file)))])
        num_of_img = X_test.shape[0]
        img_size = (X_test.shape[1], X_test.shape[2])
        Zi = model.classify(X_test)
        Y_pred = recompose(Zi, num_of_img, img_size,16)

        save_image(np.repeat(Y_pred[0][:, :, np.newaxis], 3, axis=2), test_masks_dir + "mask_" + str(img_id) + ".png")

        k += 1
        sys.stdout.write("\rProgress: " + str(k * 100 // numfiles) + "%")

    print("\rProgress: done.")
