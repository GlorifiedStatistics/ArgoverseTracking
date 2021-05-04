"""
Dataset statistics
"""
from utils.Utils import *


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
            if key in ['num_lanes', 'size']:
                continue
            d[key]['range'] = d[key]['max'] - d[key]['min']

    return stats


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
        - 'speed_raw': the speed of velocity values (sqrt(x^2 + y^2) of velocity)
        - 'speed_step': the step of speed_raw

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

    sizes = []

    for folder_path in [TRAINING_PATH, VALIDATION_PATH]:
        print("Reading in data:", folder_path)
        files = os.listdir(folder_path)
        sizes.append(len(files))

        # Build the lists to store the data
        p_raw, v_raw, p_step, v_step, p_off, v_off, speed, speed_step, speed_start = [], [], [], [], [], [], [], [], []
        p_in_raw, p_out_raw, p_in_step, p_out_step, v_in_raw, v_out_raw, v_in_step, v_out_step = [], [], [], [], [], [], [], []
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

            s = np.zeros(v.shape)
            s[:, :, 0] = np.sum(v**2, axis=2) ** 0.5

            p_raw.append(p.copy())
            v_raw.append(v.copy())
            speed.append(s.copy())
            p_in_raw.append(p[:19].copy())
            p_out_raw.append(p[19:].copy())
            v_in_raw.append(v[:19].copy())
            v_out_raw.append(v[19:].copy())

            p_init = p[0, 0, :].copy()

            # Build the steps and offsets
            p[1:] -= p[:-1]
            v[1:] -= v[:-1]
            p[0, :, :] -= p_init
            v[0, :, :] -= v[0, 0, :]

            s[1:, :, :] -= s[:-1, :, :]

            # Only do the time steps that are not offsets, and offsets that are not initial position
            p_step.append(p[1:, :, :].copy())
            v_step.append(v[1:, :, :].copy())
            p_off.append(p[0, 1:, :].copy())
            v_off.append(v[0, 1:, :].copy())
            speed_step.append(s[1:, :, :].copy())
            speed_start.append(s[0, :, :].copy())
            p_in_step.append(p[:19].copy())
            p_out_step.append(p[19:].copy())
            v_in_step.append(v[:19].copy())
            v_out_step.append(v[19:].copy())

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
                   'lane_raw': lane_raw, 'lane_step': lane_step, 'lane_norm': lane_norm, 'num_lanes': num_lanes,
                   'speed': speed, 'speed_step': speed_step, 'speed_start': speed_step, 'p_in_step': p_in_step,
                   'v_in_step': v_in_step, 'p_in_raw': p_in_raw, 'v_in_raw': v_in_raw}
        if folder_path == TRAINING_PATH:
            vals[k].update({'p_out_step': p_out_step, 'v_out_step': v_out_step, 'p_out_raw': p_out_raw,
                            'v_out_raw': v_out_raw})

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
        if k in vals['val']:
            vals['train'][k] = [vals['train'][k], vals['val'][k]]
        else:
            vals['train'][k] = [vals['train'][k]]

    ret['all'] = _make_stats(vals['train'])
    ret['all']['num_lanes'] = np.zeros([max(len(ret['all']['num_lanes'][0]), len(ret['all']['num_lanes'][1]))])
    for a in [vals['train']['num_lanes'][i] for i in range(2)]:
        ret['all']['num_lanes'][:len(a)] += a

    ret['train']['size'] = sizes[0]
    ret['val']['size'] = sizes[1]
    ret['all']['size'] = sum(sizes)

    if save_stats:
        with open(DATASET_STATS_PATH, 'wb') as f:
            pickle.dump(ret, f)

    print("Stats done!")

    return ret