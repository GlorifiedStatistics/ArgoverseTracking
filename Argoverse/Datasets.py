"""
Load in and clean data for datasets
"""
from utils.Utils import *

_NAMES = ['p', 'v', 's', 'lane', 'agent_pos']


def load_dataset(p=True, v=True, s=False, lanes=False, agent_pos=False, all_data=False, fill_data=True,
                 max_lanes=100000, train=True, val=True, flatten=False, max_data=None, redo=False, depth=None):
    """
    Loads the dataset. Combines all of the data if that has not yet been done.
    :param p: whether or not to load the position data
    :param v: whether or not to load the velocity data
    :param s: whether or not to load the speed data
    :param lanes: whether or not to load the lane data
    :param agent_pos: whether or not to load the initial agent positions data
    :param all_data: if True, loads all data regardless of other parameters
    :param fill_data: if True, fills all of the [p, v, s] data so there are 60 cars each time step (fills empty
        cars with all 0's), and makes the lane data a (num_datapoints, max_lanes) array with empty lanes filled with
        0's as well. Then, combines all of this data together so d['p'] is a now the numpy array of all data, and
        d['scene_idx'] is the full list of scene indices that is the same across all datas.

        if False, returns all of the data as it is read in from files
    :param max_lanes: if None, then is set to the largest number of lanes in any scene. Otherwise is an integer for
        max number of lanes to keep from each scene, ordered by closest to initial agent position
    :param train: whether or not to load the training data
    :param val: whether or not to load the validation data
    :param flatten: if True, flattens all of the incoming data so each data is a 1d array (or 2d if fill_data is True)
    :param max_data: the maximum number of examples to choose at random
    :param redo: if True, combines all of the data again no matter what
    :param depth: SHOULD NOT BE USED BY THE CALLER, is used to keep track of whether or not we have attempted to load
        data so we don't fall into an infinite loop if something doesn't load correctly

    :return: a list of all of the different datas loaded (train and/or val), each as a dictionary
    """
    if depth is not None:
        raise FileNotFoundError()

    print("Loading data...")

    if redo:
        print("Redoing data...")
        combine_data()

    _dict_inputs = [p, v, s, lanes, agent_pos]

    # In case some idiot doesn't want any data
    if not any(_dict_inputs):
        return None

    try:
        def _load(name, _train):
            with open(_get_data_path(name, _train), 'rb') as f:
                return pickle.load(f)

        ret = []
        _ml = min(max_lanes, 1899)
        for is_training_path in [p for p, b in zip([True, False], [train, val]) if b]:
            loads = _NAMES if all_data else [n if b else None for n, b in zip(_NAMES, _dict_inputs)]
            d = {n: _load(n, is_training_path) for n in loads if n is not None}

            _d_main_keys = list(d.keys())
            _big_keys = list(d[_d_main_keys[0]].keys())
            if "name" in _big_keys:
                _big_keys.remove("name")
            np.random.shuffle(_big_keys)
            _keys = _big_keys[:max_data] if max_data is not None else _big_keys

            # Fill the data if we need to
            if fill_data:

                # Change all the p, v, and s data
                for k in [k for k in ['p', 'v', 's'] if k in d.keys()]:
                    shape = d[k][_keys[0]].shape[-1] if len(d[k][_keys[0]].shape) > 2 else 1
                    new_data = np.zeros([len(_keys), 49 if is_training_path else 19, 60, shape])

                    for i, scene_idx in enumerate(_keys):
                        num_cars = d[k][scene_idx].shape[1]
                        new_data[i, :, :num_cars, :] = d[k][scene_idx].reshape(
                            [49 if is_training_path else 19, num_cars, shape])

                    d[k] = new_data.reshape([len(_keys), -1]) if flatten else new_data

                # Fill the lanes
                if 'lane' in d.keys():
                    new_lane = np.zeros([len(_keys), 4 * _ml])

                    for i, k in enumerate(_keys):
                        _len = len(d['lane'][k]['lane'])
                        new_lane[i, :_len * 2] = d['lane'][k]['lane'].reshape([-1])
                        new_lane[i, _ml * 2:_ml * 2 + _len * 2] = d['lane'][k]['lane_norm'].reshape([-1])

                    d['lane'] = new_lane

                # Set the new scene_idx's
                d['scene_idx'] = np.array(_keys)
                if 'agent_pos' in d.keys():
                    d['agent_pos'] = np.array([d['agent_pos'][k] for k in _keys])

            # If we are not filling, we still have to do max_data things.
            if max_data is not None and not fill_data:

                # Do things faster
                small = max_data < len(d[_d_main_keys[0]].keys()) * 0.45
                for k in _d_main_keys:
                    if small:  # Experimentally, this is the best cutoff
                        d[k] = {_k: d[k][_k] for _k in _keys}
                    else:
                        for _bk in _big_keys[max_data:]:
                            del d[k][_bk]

            ret.append(d)

        print("Loaded!")
        return ret

    except FileNotFoundError:
        try:
            print("Couldn't load data, remaking it...")
            combine_data()
            return load_dataset(*locals())  # Kinda a hack but a good one at that
        except FileNotFoundError:
            raise FileNotFoundError("Error: Could not load the data")


def _get_data_path(name, train):
    return os.path.join(CLEAN_TRAINING_PATH if train else CLEAN_VALIDATION_PATH, name + ".pkl")


def combine_data():
    """
    Combines all the data into a single file for each section of data. Sections are:
        - p: the position values
        - v: the velocity values
        - s: the speed values
        - agent_pos: the agent initial positions
        - lanes: the lane information

    Does this in two different folders: one for training and one for validation
    Each one of these files is a dictionary with keys being the scene indices, and values the data
    All of the data is in agent coordinates by default
    """
    print("Combining datasets...")

    for folder_path in [TRAINING_PATH, VALIDATION_PATH]:
        print("Reading in data:", folder_path)
        files = os.listdir(folder_path)

        all_p, all_v, all_s, all_lanes, all_agent_pos = {}, {}, {}, {}, {}
        _dicts = [all_p, all_v, all_s, all_lanes, all_agent_pos]
        for d, n in zip(_dicts, _NAMES):
            d['name'] = n

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

            # Swap the axes, and set agent car to first index
            p[[0, a_idx], :, :] = p[[a_idx, 0], :, :]
            v[[0, a_idx], :, :] = v[[a_idx, 0], :, :]
            p = np.swapaxes(p, 0, 1)[:, :num_cars, :]
            v = np.swapaxes(v, 0, 1)[:, :num_cars, :]

            agent_pos = p[0, 0, :].copy()
            p -= agent_pos

            s = np.sum(v**2, axis=2) ** 0.5

            lane = data['lane'][:, :2] - agent_pos
            lane_norm = data['lane_norm'][:, :2]

            _data = [p, v, s, {'lane': lane, 'lane_norm': lane_norm}, agent_pos]
            for _d, _dict in zip(_data, _dicts):
                _dict[data['scene_idx']] = _d

        print("Saving data...")
        for d, name in zip(_dicts, _NAMES):
            with open(_get_data_path(name, folder_path == TRAINING_PATH), 'wb') as f:
                pickle.dump(d, f)

    print("Datasets cleaned!")
