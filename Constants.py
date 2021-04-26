import cv2
import numpy as np
import pandas as pd
import os
import pickle
import progressbar
import matplotlib.pyplot as plt

RANDOM_STATE = 123456

# Filepath constants
BASE_PATH = "D:\\BigData\\CSE151B_Kaggle"
RAW_DATA_PATH = BASE_PATH + "\\raw"
TRAINING_PATH = RAW_DATA_PATH + "\\new_train\\new_train"
VALIDATION_PATH = RAW_DATA_PATH + "\\new_val_in\\new_val_in"
CLEAN_DATA_PATH = BASE_PATH + "\\clean_data"
SCENE_INDICES_PATH = BASE_PATH + "\\scene_indices.pkl"
TRACK_IDS_PATH = BASE_PATH + "\\track_ids.pkl"
PREDICTIONS_PATH = BASE_PATH + "\\predictions"
DATASET_STATS_PATH = BASE_PATH + "\\dataset_stats.pkl"
VAL_LABELS_PATH = CLEAN_DATA_PATH + "\\val_labels.npy"

# Data cleaning constants
MIN_NORMALIZE_VAL = -100
MAX_NORMALIZE_VAL = 100
MAX_LANES = 100
# The max and min values to feed into neural network
V_MAX = 2
V_MIN = -2
STD_MIN = -1
STD_MAX = 1
STRETCH_BOUNDS = True
V_RANGE = V_MAX - V_MIN
STD_STEP_DEVIATIONS = 2

# Stats constants
NUM_STATS_BINS = 10_000

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

NUMPY_FILE_EXT = '.npy'


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
    filename = _fix_file_ext(filename)
    return np.load(filename)


def save_val_labels(labels):
    np.save(_fix_file_ext(VAL_LABELS_PATH), labels)


def load_val_labels():
    return np.load(_fix_file_ext(VAL_LABELS_PATH))


def save_predictions(pred, model_name, norm_func, y_output):
    """
    Saves predictions to csv file for upload to kaggle
    :param pred: (3200, 60) numpy array of outputs x1, y1, x2, y2, ... for each scene_id

        WARNING: REMEMBER TO NOT CHANGE THE VALIDATION DATASET ROWS AROUND

    :param model_name: the classname of the model
    :param norm_func: THE ACTUAL FUNCTION, not strings
    :param y_output: the y_output type
    """
    filename = os.path.join(PREDICTIONS_PATH, model_name + "_" + norm_func.__name__ + "_" + y_output)
    labels = load_val_labels()
    vs = [('V%d' % i) for i in range(1, 61)]
    _id = "ID"
    df = pd.DataFrame(pred, columns=vs)
    df[_id] = labels
    df = df[[_id] + vs]

    if not os.path.exists(PREDICTIONS_PATH):
        os.mkdir(PREDICTIONS_PATH)

    df.to_csv(os.path.join(PREDICTIONS_PATH, _fix_file_ext(filename, '.csv')), index=False)
