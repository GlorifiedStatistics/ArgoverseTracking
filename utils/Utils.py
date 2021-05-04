import cv2
import numpy as np
import pandas as pd
import os
import pickle
import progressbar
import matplotlib.pyplot as plt
from utils.Constants import *

# Make a new print function
_PRINT = print


def _print(*args):
    """
    My own print statement in case I ever want to log things as well
    :param args: args to print
    """
    _PRINT(*args)


# Change builtin print in case I want to log to file or anything
print = _print


def _fix_file_ext(filename, ext=NUMPY_FILE_EXT):
    if '.' in filename:
        filename = filename[:-filename[::-1].index('.') - 1]
    return filename + ext


def save_numpy(arr, filename):
    filename = _fix_file_ext(filename)
    if os.path.exists(filename):
        os.remove(filename)
    np.save(filename, arr)


def load_numpy(filename):
    return np.load(_fix_file_ext(filename), allow_pickle=True)


def save_predictions(pred, name, labels):
    """
    Saves predictions to csv file for upload to kaggle
    :param pred: (3200, 60) numpy array of outputs x1, y1, x2, y2, ... for each scene_id

        WARNING: REMEMBER TO NOT CHANGE THE VALIDATION DATASET ROWS AROUND

    :param name: the name of the model and its parameters
    :param labels: the validation labels
    """
    filename = os.path.join(PREDICTIONS_PATH, name + ".csv")
    vs = [('V%d' % i) for i in range(1, 61)]
    _id = "ID"
    df = pd.DataFrame(pred, columns=vs)
    df[_id] = labels
    df = df[[_id] + vs]

    if not os.path.exists(PREDICTIONS_PATH):
        os.mkdir(PREDICTIONS_PATH)

    df.to_csv(filename, index=False)