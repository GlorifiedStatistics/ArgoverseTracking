"""
Package to load in and clean data for datasets

Raw data description:

The raw data is in two folders: one for training, and one for validation. Inside each folder is many pickle (.pkl)
    files. Each pickle file is a dictionary containing information on one individual scene.

Each scene consists of the following information:
    - Positions of other objects in the scene (always 60 total, even if they aren't used. Unused values are filled
        with 0's). These are sampled at 10HZ.
    - Velocities of other objects in the scene (always 60 total, even if they aren't used. Unused values are filled
        with 0's). These are sampled at 10HZ.
    - An ID for each vehicle.
    - The ID of the vehicle that we are tracking (the 'agent' vehicle).
    - Information on where the lanes are. These are positions of the centers of the lanes, and what direction those
        lanes are pointing (the lane normals). There are an arbitrary number of lanes for each scene.

Each pickle file contains a dictionary with the following fields:
    - 'city': a string for the city the data was taken from. Can either be 'PIT' or 'MIA'
    - 'scene_idx': a unique non-zero integer acting as an ID for the current scene (unique across the union of training
        and validation sets)
    - 'agent_id': a string of the form "00000000-0000-0000-0000-0000XXXXXXXX" with the X's being digits that identifies
        the 'agent' we want to track (predict future movement of) in this scene. This ID is always 36 characters long
        (32 digits if you don't include the -'s)
    - 'car_mask': a (60, 1) numpy array of float 1's and 0's. Each row corresponds to the row index of each car in
        the scene (IE: this matches with the track_id, p_in, p_out, v_in, and v_out first dimensions). There is a '1'
        in the row if the values in the first dimension of track_id, p_in, etc. correspond to an actual object in the
        scene, or are just empty because there were less than 60 objects in the scene. The total number of 1's in this
        mask is the total number of objects in the scene, including the agent car we are tracking. This allows us to
        quickly get all of the values in track_id, p_in, etc. that correspond to actual objects like so:

        actual_objects = track_id[car_mask.reshape([-1]).astype(int)]

        NOTE: the values in the numpy array are floats, and thus need to be converted to integers before using them
        as a mask (hence the .astype(int)

        The car_mask is laid out such that the only 1's occur at the beginning of the array. So, in the event you had
        10 objects in the scene: car_mask[:10, 1] == 1, and car_mask[:10, 1] == 0. This means all of the actually
        tracked objects in each of track_id, p_in, etc come first, and the rest of the values are dummy values.
    - 'track_id': a (60, 30, 1) numpy array of strings. Each string is of the form
        "00000000-0000-0000-0000-0000XXXXXXXX" with the X's being digits. This describes a unique id (unique to the
        scene, not all of the data) for each object in the scene. The whole array is 60 "30 by 1 column vectors" where
        every one of the 30 elements in the column vector are the same track_id string. Don't ask me why this is the
        case, I have no idea. They could have just as easily made a 1-D list of 60 track_ids and accomplished the same
        thing. Honestly, whoever made this was probably just too lazy to change it, and I don't blame them; I'd
        probably do the same thing.

        This array follows the same indexing as the 'car_mask'. So, the first n 1's in the car mask array describe
        the track_ids for all of the actually tracked objects in the scene. After the first n actually tracked objects,
        the track_id changes to 'dummyK' where K is the integer index of the dummy variable starting at 0 (IE: 'dummy0',
        'dummy1', ..., 'dummy[60-n-1]').
    - 'p_in': a (60, 19, 2) numpy array of floats. These are the x,y coordinates of each tracked object
        in the scene, for a max of 60 objects. What those coordinates are relative to I do not know (probably some
        longitude/latitude in or near whatever city the data is from), so you may not want to use the exact values
        in your project, and instead care only about the change in position at each time step. There are 19 samples
        per object meaning the data is probably only for 1.9 seconds, not 2 seconds exactly. This array follows the
        same indexing as car_mask, so the first n arrays in p_in correspond to the n tracked objects in the scene,
        and the following 60-n arrays are filled with all 0's.

        NOTE: some values may be 0 or negative even if we are tracking them, indicating the object is at the very
        edge of the coordinate system
    - 'v_in': same as 'p_in', but for tracking velocities instead of position. These values can be positive, negative,
        or 0. Again, I don't know what the units are. It might be on the Argoverse website somewhere?
        NOTE: some values may still be 0.0 for velocity even if it is for an object we are tracking (IE: the object is
        not moving)
    - 'p_out': same as 'p_in', but is instead of shape (60, 30, 2). These are the positions your model should learn
        to predict.
        NOTE: the validation sets do not contain this key as this is what your model should predict.
    - 'v_out': same as 'v_in', but is instead of shape (60, 30, 2). You do not need to predict these for the final
        project. You might want to discard them, unless you want to instead have your model predict velocity instead
        of position, and you can calculate the final positions by adding some small multiple of the velocity?
        NOTE: the validation sets do not contain this key
        NOTE: some values may still be 0.0 for velocity even if it is for an object we are tracking (IE: the object is
        not moving)
    - 'lane': a (k, 3) numpy array of floats where 'k' is the number of lanes in the scene. There can be a
        different number of lanes for each scene, so there is no guarantee on the size of 'k'. These describe the x,y,z
        coordinates of the center of lane nodes (where driving lanes are in the scene). For some reason, the
        z-coordinate is included, but is always 0, so you can ignore it and just look at the x,y. These seem to be
        in the same coordinate system as the p_in and p_out coordinates. Perhaps you want to change these to be
        relative to something else in the scene?
        NOTE: some x,y values may be 0 or negative indicating the center of the lane is at the very edge of the
        coordinate system
    - 'lane_norm': a (k, 3) numpy array of floats where 'k' is the number of lanes in the scene (same size as the 'k'
        in 'lane'). These describe the x,y,z vector direction of the corresponding lane normal (the direction the
        lane center with the same index is pointing). These values can be positive or negative, but never 0 (for x,y).
        Again, the z direction is always 0 and can be ignored, and the coordinate system seems to be the same as p_in.

    QUESTIONS FOR DATA:
        - Why do all track_id's start with all 0's?
"""

from Constants import *
from sklearn.model_selection import train_test_split as tts

_T = (TRAINING_PATH, VALIDATION_PATH)
_V = (VALIDATION_PATH, TRAINING_PATH)


class NormFuncs:
    @staticmethod
    def noop(arr, _ss):
        return arr

    @staticmethod
    def inv_noop(arr, _ss):
        return arr

    @staticmethod
    def linear(arr, _ss):
        return ((arr - _ss['_min']) / _ss['_range']) * V_RANGE + V_MIN

    @staticmethod
    def inv_linear(arr, _ss):
        return ((arr - V_MIN) / V_RANGE) * _ss['_range'] + _ss['_min']

    @staticmethod
    def std_step(arr, _ss):
        ret = arr.copy()

        old_shape = arr.shape
        ret = ret.reshape([-1, 2])

        for i in range(2):
            std_min = min(100000 if STRETCH_BOUNDS else STD_MIN, _ss['mean'][i] - STD_STEP_DEVIATIONS * _ss['std'][i])
            std_max = max(-100000 if STRETCH_BOUNDS else STD_MAX, _ss['mean'][i] + STD_STEP_DEVIATIONS * _ss['std'][i])
            mins = np.argwhere(arr[:, i] < std_min).reshape([-1])
            maxs = np.argwhere(arr[:, i] > std_max).reshape([-1])
            stds = np.argwhere(np.logical_and(arr[:, i] >= std_min, arr[:, i] <= std_max)).reshape([-1])

            ret[:, i][mins] = (ret[mins][:, i] - _ss['_min'][i]) / (std_min - _ss['_min'][i]) * (STD_MIN - V_MIN) + V_MIN
            ret[:, i][maxs] = (ret[maxs][:, i] - std_max) / (_ss['_max'][i] - std_max) * (V_MAX - STD_MAX) + STD_MAX
            ret[:, i][stds] = (ret[stds][:, i] - std_min) / (std_max - std_min) * (STD_MAX - STD_MIN) + STD_MIN

        return ret.reshape(old_shape)

    @staticmethod
    def inv_std_step(arr, _ss):
        ret = arr.copy()

        old_shape = arr.shape
        ret = ret.reshape([-1, 2])

        for i in range(2):
            std_min = min(10000 if STRETCH_BOUNDS else STD_MIN, _ss['mean'][i] - STD_STEP_DEVIATIONS * _ss['std'][i])
            std_max = max(-10000 if STRETCH_BOUNDS else STD_MAX, _ss['mean'][i] + STD_STEP_DEVIATIONS * _ss['std'][i])
            mins = np.argwhere(arr[:, i] < STD_MIN).reshape([-1])
            maxs = np.argwhere(arr[:, i] > STD_MAX).reshape([-1])
            stds = np.argwhere(np.logical_and(arr[:, i] >= STD_MIN, arr[:, i] <= STD_MAX)).reshape([-1])

            ret[:, i][mins] = ((ret[mins][:, i] - V_MIN) / (STD_MIN - V_MIN)) * (std_min - _ss['_min'][i]) + _ss['min'][i]
            ret[:, i][maxs] = ((ret[maxs][:, i] - STD_MAX) / (V_MAX - STD_MAX)) * (_ss['_max'][i] - std_max) + std_max
            ret[:, i][stds] = ((ret[stds][:, i] - STD_MIN) / (STD_MAX - STD_MIN)) * (std_max - std_min) + std_min

        return ret.reshape(old_shape)

    @staticmethod
    def tanh(arr, _ss):
        return np.tanh(arr)

    @staticmethod
    def inv_tanh(arr, _ss):
        return np.arctanh(arr)

    @staticmethod
    def get_norm_funcs(norm_func):
        if isinstance(norm_func, str):
            norm_func = norm_func.lower()
            if norm_func in ['noop', 'none', 'no']:
                norm_func = NormFuncs.noop
                inv_norm_func = NormFuncs.noop
            elif norm_func in ['linear', 'lin', 'minmax', 'mm', 'min_max', 'min-max']:
                norm_func = NormFuncs.linear
                inv_norm_func = NormFuncs.inv_linear
            elif norm_func in ['tanh']:
                norm_func = NormFuncs.tanh
                inv_norm_func = NormFuncs.inv_tanh
            elif norm_func in ['std', 'std_step']:
                norm_func = NormFuncs.std_step
                inv_norm_func = NormFuncs.inv_std_step
            else:
                raise ValueError("Unknown norm_func: %s" % norm_func)

        elif norm_func is None:
            norm_func = NormFuncs.noop
            inv_norm_func = NormFuncs.noop

        else:
            inv_norm_func = NormFuncs().__getattribute__("inv_" + norm_func.__name__)

        return norm_func, inv_norm_func


def load_dataset(redo=False, norm_func=NormFuncs.linear, include_val=False, y_output='full'):
    """
    Loads the cleaned dataset.
    :return: a 4-tuple or 6-tuple of train_x, train_y, test_x, test_y, val_x, val_y
    """
    print("Loading dataset...")

    norm_func, inv_norm_func = NormFuncs.get_norm_funcs(norm_func)

    stats = load_stats()

    def new_inv_norm_func(arr):
        return inv_norm_func(arr, stats['all']['p_step'])

    if y_output not in ['single_step', 'full']:
        raise ValueError("Unknown y_output: %s" % y_output)

    filename = os.path.join(CLEAN_DATA_PATH, norm_func.__name__ + "_" + y_output)

    if redo:
        print("Redo-ing data...")
        clean_dataset(norm_func=norm_func, y_output=y_output)

    def _ret():
        train_x, train_y = load_numpy(filename + "_train_x"), load_numpy(filename + "_train_y")
        test_x, test_y = load_numpy(filename + "_test_x"), load_numpy(filename + "_test_y")
        test_pred_off = load_numpy(filename + "_test_pred_off")
        ret = {'train_x': train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y,
               'test_pred_off': test_pred_off, 'inv_norm_func': new_inv_norm_func}

        if include_val:
            val_x, val_pred_off = load_numpy(filename + "_val_x"), load_numpy(filename + "_val_pred_off")
            ret.update({'val_x': val_x, 'val_pred_off': val_pred_off})

        return ret

    try:
        return _ret()
    except FileNotFoundError:
        print("Could not find cleaned data files, redo-ing...")
        clean_dataset(norm_func=norm_func, y_output=y_output)
        try:
            return _ret()
        except FileNotFoundError as e:
            print("Error reading in cleaned data files:", e)


def load_stats(redo=False):
    """
    Loads the dataset statistics. See generate_dataset_statistics() for the dictionary that is returned
    :param redo: if True, calls generate_dataset_statistics() no matter what to redo the stats
    :return: a dictionary
    """
    if redo:
        print("redo is True, remaking stats...")
        generate_dataset_statistics()
        return load_stats(redo=False)

    print("Loading dataset statistics...")

    try:
        with open(DATASET_STATS_PATH, 'rb') as f:
            stats = pickle.load(f)
    except FileNotFoundError:
        print("Could not find stats file, generating again...")
        generate_dataset_statistics()
        try:
            with open(DATASET_STATS_PATH, 'rb') as f:
                stats = pickle.load(f)
        except:
            raise FileNotFoundError("Could not load in the file, even after generating it...")

    print("Statistics loaded!")

    # Do this to make future calls easier (normalizing in clean_dataset)
    # We want this here in case I wish to change MIN_NORMALIZE_VAL etc.
    for k, d in stats.items():
        for key in d.keys():
            if key in ['num_lanes']:
                continue
            d[key]['_min'] = d[key]['min'].copy()
            d[key]['_min'][d[key]['_min'] < MIN_NORMALIZE_VAL] = MIN_NORMALIZE_VAL
            d[key]['_max'] = d[key]['max'].copy()
            d[key]['_max'][d[key]['_max'] > MAX_NORMALIZE_VAL] = MAX_NORMALIZE_VAL
            d[key]['_range'] = d[key]['_max'] - d[key]['_min']

    return stats


def clean_dataset(num_samples=5, norm_func='linear', y_output='full'):
    """
    Cleans the dataset to be used in the RNN. Values are only saved and not returned

    :param num_samples: the number of samples to take per scene. Only used if y_output == 'single_step'
    :param norm_func: the function to normalize values so they aren't too large for networks. This is ignored
        if model_type is not 'nn'
    :param y_output: how to have the Y datasets. 'single_step' to have the model only predict the single next
        time step (and therefore use num_samples), or 'full' to predict the full 3 seconds (30 timesteps) of output.
    """
    print("Cleaning datasets...")
    norm_func, inv_norm_func = NormFuncs.get_norm_funcs(norm_func)

    if y_output not in ['full', 'single_step']:
        raise ValueError("Unknown y_output: %s" % y_output)

    stats = load_stats()

    if num_samples < 1 or num_samples > 29:
        raise ValueError("Num samples is not good: %d", num_samples)

    for folder_path in _T:
        print("Reading in data:", folder_path)
        files = os.listdir(folder_path)

        num_examples = (len(files) * num_samples) if folder_path == TRAINING_PATH and y_output == 'single_step' else len(files)
        X = np.empty([num_examples, 19 * 60 * 4 + 4 * MAX_LANES])
        Y = np.empty([num_examples, 2]) if y_output == 'single_step' else np.empty([num_examples, 2 * 30])
        pred_off = np.empty([num_examples, 2])

        val_labels = []

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

            raw_lane_step = data['lane'][:, :2]
            raw_lane_norm = data['lane_norm'][:, :2]
            lane_cutoff = min(len(raw_lane_step), MAX_LANES)

            # Get all the data if validation, or num_samples per scene if training
            # 49-19-1 because need to include output values
            starts = np.random.choice(list(range(49 - 19 - 1)), [num_samples], replace=False) \
                if folder_path == TRAINING_PATH and y_output == 'single_step' else [0]

            idx = i * num_samples if folder_path == TRAINING_PATH and y_output == 'single_step' else i
            for s_idx, s in enumerate(starts):
                # Do p data first, shift initial time step positions based on position of agent car, save the
                #   current agent car pos, change p to be delta positions instead
                p_add = p[s:].copy()
                agent_pos = p_add[0, 0, :].copy()
                final_pos = p_add[:, 0, :].copy()

                p_add[1:, :num_cars, :] -= p_add[:-1, :num_cars, :]
                p_add[0, :num_cars, :] = p_add[0, :num_cars, :] - agent_pos # we want this done after changing to deltas

                # Must normalize the initial offsets differently than the rest of p
                p_add[0:1, :, :] = norm_func(p_add[0:1, :, :], stats['all']['p_off'])
                p_add[1:, :, :] = norm_func(p_add[1:, :, :], stats['all']['p_step'])

                # Make the new row with first set of elements being p_add
                arr = p_add[:19].copy().reshape([-1])

                # Now to do v stuff, copy what was done for p
                v_add = v[s:s + 19].copy()
                v_init = v_add[0, 0, :]

                v_add[1:, :num_cars, :] -= v_add[:-1, :num_cars, :]
                v_add[0, :num_cars, :] = v_init - v_add[0, :num_cars, :]  # we want this done after changing to deltas

                v_add[0:1, :, :] = norm_func(v_add[0:1, :, :], stats['all']['v_off'])
                v_add[1:, :, :] = norm_func(v_add[1:, :, :], stats['all']['v_step'])

                arr = np.append(arr, v_add.copy().reshape([-1]))

                # Now for lane thingies
                t_lane_step = raw_lane_step - agent_pos
                closest = np.argsort(np.sum(t_lane_step ** 2, axis=1) ** 0.5)[:lane_cutoff]
                lane_step = np.zeros([MAX_LANES, 2])
                lane_norm = np.zeros([MAX_LANES, 2])

                lane_step[:lane_cutoff, :] = norm_func(t_lane_step[closest], stats['all']['lane_step'])
                lane_norm[:lane_cutoff, :] = norm_func(raw_lane_norm[closest], stats['all']['lane_norm'])

                arr = np.append(arr, lane_step.reshape([-1]))
                arr = np.append(arr, lane_norm.reshape([-1]))
                X[idx + s_idx] = arr

                # Make Y if we are training
                if folder_path == TRAINING_PATH:
                    if y_output == 'single_step':
                        Y[idx + s_idx] = norm_func(p_add[s+19, 0, :] - p_add[s+18, 0, :], stats['all']['p_step'])
                    elif y_output == 'full':
                        Y[idx + s_idx] = norm_func(p_add[19:, 0, :], stats['all']['p_step']).reshape([-1])

                # Make the pred_off to save starting positions
                pred_off[idx + s_idx] = final_pos[18].copy()

            # X and Y should be done for this file

        filename = os.path.join(CLEAN_DATA_PATH, norm_func.__name__ + "_" + y_output)

        if folder_path == TRAINING_PATH:
            train_x, test_x, train_y, test_y, _, test_pred_off = tts(X, Y, pred_off, test_size=0.2, random_state=RANDOM_STATE)
            save_numpy(train_x, filename + "_train_x")
            save_numpy(train_y, filename + "_train_y")
            save_numpy(test_x, filename + "_test_x")
            save_numpy(test_y, filename + "_test_y")
            save_numpy(test_pred_off, filename + "_test_pred_off")
        else:
            save_numpy(X, filename + "_val_x")
            save_numpy(pred_off, filename + "_val_pred_off")
            save_val_labels(val_labels)

    print("Datasets cleaned!")


def check_dataset_assumptions():
    """
    Used to check all the assumptions I am making about the dataset by brute force.
    Generates two pkl files: a list of ints for the scene_indices (at SCENE_INDICES_PATH), and a list of strings for
        the track_ids (at TRACK_IDS_PATH).
    """
    print("Checking assumptions...")

    _d_keys = ['p_in', 'p_out', 'v_in', 'v_out', 'lane', 'lane_norm', 'car_mask', 'scene_idx', 'agent_id', 'track_id',
               'city']
    id_start = "00000000-0000-0000-0000-0"
    scene_indices = []
    track_ids = []

    for folder_path in _T:
        d_keys = _d_keys if folder_path == TRAINING_PATH else [k for k in _d_keys if k not in ['p_out', 'v_out']]

        print("Reading in data:", folder_path)
        files = os.listdir(folder_path)

        for i in progressbar.progressbar(range(len(files))):
            file = files[i]
            with open(os.path.join(folder_path, file), 'rb') as f:
                data = pickle.load(f)

            assert len(list(data.keys())) == len(d_keys)
            for k in d_keys:
                assert k in data.keys()

            # Check city
            a = data['city']
            assert type(a) == str
            assert a in ['PIT', 'MIA']

            # Check scene_idx
            a = data['scene_idx']
            assert type(a) == int
            assert a >= 0
            scene_indices.append(a)

            # Check agent_id
            a = data['agent_id']
            assert type(a) == str
            assert len(a) == 36
            assert a.startswith(id_start)

            # Check car_mask
            a = data['car_mask']
            assert type(a) == np.ndarray
            assert len(a.shape) == 2
            assert (np.array(a.shape) == [60, 1]).all()
            assert isinstance(a[0][0], np.float32)
            a = a.astype(int)
            u, c = np.unique(a, return_counts=True)
            assert u[0] == 0
            assert u[1] == 1
            num_cars = c[1]
            assert (a[:num_cars]).all()
            assert not a[num_cars:].any()

            # Check track_id
            a = data['track_id']
            assert type(a) == np.ndarray
            assert len(a.shape) == 3
            assert (np.array(a.shape) == [60, 30, 1]).all()

            for i, r in enumerate(a):
                us = np.unique(r)
                assert len(us) == 1
                id = us[0]
                assert type(id) == str

                if i == 0:
                    assert id == "00000000-0000-0000-0000-000000000000"
                elif i < num_cars:
                    assert len(id) == 36
                    assert id.startswith(id_start)
                    track_ids.append(id)
                else:
                    assert id == 'dummy%d' % (i - num_cars)

            # Check p_in, p_out, v_in, v_out
            _kl = ['p_in', 'p_out', 'v_in', 'v_out'] if folder_path == TRAINING_PATH else ['p_in', 'v_in']
            for key in _kl:
                a = data[key]
                assert len(a.shape) == 3
                if key[-3:] == 'out':
                    assert (np.array(a.shape) == [60, 30, 2]).all()
                else:
                    assert (np.array(a.shape) == [60, 19, 2]).all()

                if key[0] == 'p':
                    assert (a[:num_cars, :, :] != 0).all()
                    assert (a[num_cars:, :, :] == 0).all()

            # Check lane
            a = data['lane']
            assert len(a.shape) == 2
            assert a.shape[1] == 3
            assert a.shape[0] != 0
            assert (a[:, 2] == 0).all()
            lane_size = a.shape[0]

            # Check lane norm
            a = data['lane_norm']
            assert len(a.shape) == 2
            assert a.shape[1] == 3
            assert a.shape[0] != 0
            assert (a[:, 2] == 0).all()
            assert (a[:, :2] != 0).all()
            assert a.shape[0] == lane_size

        # Do some post checks
        print("Writing scene_indices and track_ids to file...")
        with open(SCENE_INDICES_PATH, 'wb') as f:
            pickle.dump(scene_indices, f)
        with open(TRACK_IDS_PATH, 'wb') as f:
            pickle.dump(track_ids, f)

        print("Doing some post-assumptions...")
        if len(np.unique(scene_indices)) != len(scene_indices):
            print("SCENE INDICES ARE NOT UNIQUE")
        if len(np.unique(track_ids)) != len(track_ids):
            print("TRACK_IDS ARE NOT UNIQUE")

        print("All done!")


def generate_dataset_statistics(save_stats=True):
    """
    Used to generate some statistics about the datasets
    :param save_stats: whether or not to save the statistics to the DATASET_STATS pkl file
    :return: a dictionary of:
        - 'train': a dictionary of values on training set
        - 'val': one for validation
        - 'all': one for both train and val combined

        Each sub-dictionary is a set of statistics about their respective dataset. These statistics are a dictionary of:
        - 'p_raw': the raw position values, which are appended on to each other in the case of training data (p_in, p_out)
        - 'v_raw': same as p_raw, but for velocity values
        - 'p_step': the position values as increments from previous values, starting at time step t=1 (the initial is
            dropped as there is nothing for it to increment from)
        - 'v_step': same as p_step, but for velocity
        - 'p_off': the offset of initial positions of all objects in the scene from the agent car (the agent car is
            ignored as it's value would always be 0)
        - 'v_off': same as p_off, but for velocity
        - 'lane_raw': the raw x,y position values for lane center nodes
        - 'lane_step': the offsets of the lane center nodes from the initial position of the agent car
        - 'lane_norm': the raw normals of the lane center nodes

        Each set of statistics is a dictionary with the following statistics:
        - 'min': min value of x and y columns
        - 'max': max value of x and y columns
        - 'mean': average value of x and y columns
        - 'std': standard deviation of x and y columns
        - 'absolute_min': values closest to 0 of x and y columns
        - 'hist_x': histogram of x column, raw output of np.histogram(col, bins=NUM_STATS_BINS)
        - 'hist_y': same as hist_x, but for y column
    """
    vals = {}

    for folder_path in _T:
        print("Reading in data:", folder_path)
        files = os.listdir(folder_path)

        # Build the lists to store the data
        p_raw, v_raw, p_step, v_step, p_off, v_off = [], [], [], [], [], []
        lane_raw, lane_step, lane_norm = [], [], []
        num_lanes = np.zeros([10])

        for i in progressbar.progressbar(range(len(files))):
            file = files[i]
            with open(os.path.join(folder_path, file), 'rb') as f:
                data = pickle.load(f)

            # The number of cars in the scene, and the agent index in the arrays
            car_idx = int(sum(data['car_mask'].astype(int)))
            a_idx = list(data['track_id'][:, 0, 0].reshape([-1])).index(data['agent_id'])

            # Get the positions and velocities of cars
            if folder_path == TRAINING_PATH:
                p = np.append(data['p_in'], data['p_out'], axis=1)
                v = np.append(data['v_in'], data['v_out'], axis=1)
            else:
                p = data['p_in']
                v = data['v_in']

            # Swap the axes, only use the tracked cars, and set agent car to first index
            p[[0, a_idx], :, :] = p[[a_idx, 0], :, :]
            v[[0, a_idx], :, :] = v[[a_idx, 0], :, :]
            p = np.swapaxes(p[:car_idx, :, :], 0, 1)
            v = np.swapaxes(v[:car_idx, :, :], 0, 1)

            p_raw.append(p.copy())
            v_raw.append(v.copy())

            p_init = p[0, 0, :].copy()

            # Build the steps and offsets
            p[1:] -= p[:-1]
            v[1:] -= v[:-1]
            p[0, :, :] -= p_init
            v[0, :, :] -= v[0, 0, :]

            # Only do the time steps that are not offsets, and offsets that are not initial position
            p_step.append(p[1:, :, :].copy())
            v_step.append(v[1:, :, :].copy())
            p_off.append(p[0, 1:, :].copy())
            v_off.append(v[0, 1:, :].copy())

            # Do lane stuff
            lane_raw.append(data['lane'][:, :2].copy())
            lane_step.append((data['lane'][:, :2] - p_init).copy())
            lane_norm.append(data['lane_norm'][:, :2].copy())

            if len(num_lanes) <= len(data['lane']):
                temp = num_lanes.copy()
                num_lanes = np.zeros([len(data['lane']) + 1])
                num_lanes[:len(temp)] += temp
            num_lanes[len(data['lane'])] += 1

        k = 'train' if folder_path == TRAINING_PATH else 'val'
        vals[k] = {'p_raw': p_raw, 'v_raw': v_raw, 'p_step': p_step, 'v_step': v_step, 'p_off': p_off, 'v_off': v_off,
                   'lane_raw': lane_raw, 'lane_step': lane_step, 'lane_norm': lane_norm, 'num_lanes': num_lanes}

    # Now to do the actual stats
    print("Doing stats...")

    def _stats(arr):
        return [np.min(arr, axis=0), np.max(arr, axis=0), np.mean(arr, axis=0), np.std(arr, axis=0),
                np.min(abs(arr), axis=0), np.histogram(arr[:, 0], bins=NUM_STATS_BINS),
                np.histogram(arr[:, 1], bins=NUM_STATS_BINS)]

    def _make_stats(ds):
        _ret = {}
        for k in ds.keys():
            if k in ['num_lanes']:
                _ret[k] = ds[k]
                continue
            # Concatenate arrays
            for i in range(len(ds[k])):
                ds[k][i] = ds[k][i].reshape([-1, 2])
            ds[k] = np.concatenate(ds[k], axis=0)
            _ret[k] = {n: v for n, v in
                       zip(['min', 'max', 'mean', 'std', 'abs_min', 'hist_x', 'hist_y'], _stats(ds[k]))}
        return _ret

    ret = {}
    for name in vals.keys():
        ret[name] = _make_stats(vals[name])

    # Now all the arrays should be concatenated, make the 'all' data as well
    # No need to waste memory, use 'train'
    for k in vals['train'].keys():
        vals['train'][k] = [vals['train'][k], vals['val'][k]]

    ret['all'] = _make_stats(vals['train'])
    ret['all']['num_lanes'] = np.zeros([max(len(ret['all']['num_lanes'][0]), len(ret['all']['num_lanes'][1]))])
    for a in [vals['train']['num_lanes'][i] for i in range(2)]:
        ret['all']['num_lanes'][:len(a)] += a

    if save_stats:
        with open(DATASET_STATS_PATH, 'wb') as f:
            pickle.dump(ret, f)

    print("Stats done!")

    return ret
