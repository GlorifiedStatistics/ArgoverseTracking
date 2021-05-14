RANDOM_STATE = 123456

# Filepath constants
DATA_PATH = "./data"
RAW_DATA_PATH = DATA_PATH + "/raw"
TRAINING_PATH = RAW_DATA_PATH + "/new_train"
VALIDATION_PATH = RAW_DATA_PATH + "/new_val_in"
CLEAN_DATA_PATH = DATA_PATH + "/clean_data"
SCENE_INDICES_PATH = DATA_PATH + "/scene_indices.pkl"
TRACK_IDS_PATH = DATA_PATH + "/track_ids.pkl"
PREDICTIONS_PATH = DATA_PATH + "/predictions"
DATASET_STATS_PATH = DATA_PATH + "/dataset_stats.pkl"
MODELS_PATH = DATA_PATH + "/models"

NUMPY_FILE_EXT = '.npy'
MODEL_EXT = '.mdl'

# Data cleaning constants
MAX_LANES = 100

# Stats constants
NUM_STATS_BINS = 10_000

# Miss distance for miss rate
MISS_DIST = 2
