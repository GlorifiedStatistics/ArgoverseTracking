"""
Load in and clean data for datasets
"""
from utils.Utils import *
from Argoverse.Stats import load_stats
from Argoverse.NormFuncs import get_norm_funcs
from sklearn.model_selection import train_test_split as tts


def _cdp(data_type):
    """
    _combined_data_path
    Returns the path to the combined data file for the given data_type
    :param data_type: either 'step' or 'raw'
    """
    if not os.path.exists(CLEAN_DATA_PATH):
        os.mkdir(CLEAN_DATA_PATH)
    return CLEAN_DATA_PATH + "/%s%s" % (data_type, NUMPY_FILE_EXT)


def load_dataset(norm_func=None, data_type='step', include_lanes=True, max_data=None, extra_features=False,
                 y_norm_func=None, redo=False, **kwargs):
    """
    Loads the cleaned dataset, then normalizes based on norm_func.
    :param norm_func: the function to normalize data
    :param data_type: the type of the data. 'step' if we want the data to be step values, 'raw' for raw values
    :param include_lanes: whether or not to include lane information
    :param max_data: the max number of rows in both train and test sets
    :param extra_features: if True, then add some extra features to the dataset (see _add_features() for the
        added features)
    :param y_norm_func: if not None, then the train_y and test_y will have use this as a normalization function
        instead of norm_func
    :param redo: whether or not to redo the data
    :return: 5-tuple of train_x, train_y, test_x, test_y, val_x, val_labels
    """
    print("Loading dataset...")

    stats = load_stats()['all']
    norm_func = get_norm_funcs(norm_func)[0]
    y_norm_func = get_norm_funcs(y_norm_func)[0] if y_norm_func is not None else norm_func

    # To easily return the data
    def l():
        with open(_cdp(data_type), 'rb') as f:
            tr_x, tr_y, tx, ty, t_po, tx_o, v, v_po, vl = pickle.load(f)

        if max_data is not None:
            tr_x = tr_x[:max_data]
            tr_y = tr_y[:max_data]
            tx = tx[:max_data]
            ty = ty[:max_data]
            t_po = t_po[:max_data]

        if not include_lanes and not extra_features:
            end = 19 * 60 * 4
            tr_x, tx, v = tr_x[:, :end], tx[:, :end], v[:, :end]

        elif include_lanes and not extra_features:
            end = 19 * 60 * 4 + 4 * MAX_LANES
            tr_x, tx, v = tr_x[:, :end], tx[:, :end], v[:, :end]

        elif not include_lanes and extra_features:
            e1, s1 = 19 * 60 * 4, 19 * 60 * 4 + 4 * MAX_LANES
            tr_x = np.append(tr_x[:, :e1], tr_x[:, s1:], axis=1)
            tx = np.append(tx[:, :e1], tx[:, s1:], axis=1)
            v = np.append(v[:, :e1], v[:, s1:], axis=1)

        print("Normalizing data...")
        keys = ['train_x', 'train_y', 'test_x', 'test_y', 'val_x', 'test_pred_off', 'test_x_off', 'val_pred_off',
                'val_labels']
        ret = {}
        for k, a in zip(keys[:5], [tr_x, tr_y, tx, ty, v]):
            if k[-1] == 'y':
                ret[k] = y_norm_func(a, stats, **kwargs)
            else:
                ret[k] = norm_func(a, stats, **kwargs)
        ret.update({k:v for k, v in zip(keys[5:], [t_po, tx_o, v_po, vl])})
        return ret, stats

    # For checking to make sure the initial clean data file exists
    try:
        if redo:
            raise FileNotFoundError()
        return l()
    except FileNotFoundError:
        print("Could not find cleaned data files, redo-ing...")
        combine_data()
        try:
            return l()
        except FileNotFoundError as e:
            print("Error reading in cleaned data files:", e)


def combine_data(data_type='step'):
    """
    Combines all the data into a single file.
    :param data_type: the type of the data. 'step' if we want the data to be step values, 'raw' for raw values
    """
    print("Cleaning datasets...")

    out = None
    val_labels = []

    for folder_path in [TRAINING_PATH, VALIDATION_PATH]:
        print("Reading in data:", folder_path)
        files = os.listdir(folder_path)

        X = np.empty([len(files), 19 * 60 * 4 + 4 * MAX_LANES + 19 * 60])
        Y = np.empty([len(files), 60])
        pred_off = np.empty([len(files), 2])
        test_off = np.empty([len(files), 2])

        for i in progressbar.progressbar(range(len(files))):
            file = files[i]
            with open(os.path.join(folder_path, file), 'rb') as f:
                data = pickle.load(f)

            # The number of cars in the scene, and the agent index in the arrays
            a_idx = list(data['track_id'][:, 0, 0].reshape([-1])).index(data['agent_id'])
            num_cars = int(sum(data['car_mask']))

            # Get the positions and velocities of cars
            if folder_path == TRAINING_PATH:
                p = np.append(data['p_in'], data['p_out'], axis=1)
                v = np.append(data['v_in'], data['v_out'], axis=1)
            else:
                p = data['p_in']
                v = data['v_in']
                val_labels.append(data['scene_idx'])

            # Swap the axes, and set agent car to first index
            p[[0, a_idx], :, :] = p[[a_idx, 0], :, :]
            v[[0, a_idx], :, :] = v[[a_idx, 0], :, :]
            p = np.swapaxes(p, 0, 1)
            v = np.swapaxes(v, 0, 1)

            s = np.sum(v**2, axis=2) ** 0.5

            # Get the closest MAX_LANES lane data (closest to initial agent_pos)
            lane = data['lane'][:, :2]
            lane_norm = data['lane_norm'][:, :2]
            lane_cutoff = min(len(lane), MAX_LANES)
            lane_keeps = np.argsort(np.sum((lane - p[0, 0, :])**2, axis=1)**0.5)[:lane_cutoff]

            lane = lane[lane_keeps]
            lane_norm = lane_norm[lane_keeps]

            agent_pos = p[0, 0, :].copy()
            pred_off[i] = p[18, 0, :].copy()
            test_off[i] = p[0, 0, :].copy()

            if data_type == 'step':
                p[1:, :num_cars, :] -= p[:-1, :num_cars, :]
                p[0, :num_cars, :] -= agent_pos  # we do this after changing to deltas
                v[1:, :num_cars, :] -= v[:-1, :num_cars, :]
                v[0, :num_cars, :] -= v[0, 0, :]
                s[1:, :num_cars] -= s[:-1, :num_cars]

                lane -= agent_pos

            tl = np.zeros([MAX_LANES * 2])
            tln = np.zeros([MAX_LANES * 2])
            tl[:len(lane) * 2] = lane.reshape([-1])
            tln[:len(lane) * 2] = lane_norm.reshape([-1])
            lane, lane_norm = tl, tln

            X[i] = np.concatenate([p[:19].reshape([-1]), v[:19].reshape([-1]), lane.reshape([-1]),
                                   lane_norm.reshape([-1]), s[:19].reshape([-1])])
            if folder_path == TRAINING_PATH:
                Y[i] = p[19:, 0, :].reshape([-1])

        if folder_path == TRAINING_PATH:
            tr_x, tx, tr_y, ty, _, tpo, _, toff = tts(X, Y, pred_off, test_off, test_size=0.2,
                                                      random_state=RANDOM_STATE)
            out = [a.copy() for a in [tr_x, tr_y, tx, ty, tpo, toff]]
        else:
            out += [X.copy(), pred_off.copy(), np.array(val_labels)]

    print("Saving numpy arrays...")
    with open(_cdp(data_type), 'wb') as f:
        pickle.dump(out, f)

    print("Datasets cleaned!")