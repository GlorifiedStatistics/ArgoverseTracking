"""
Used to check dataset assertions
"""
from utils.Utils import *


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

    for folder_path in [TRAINING_PATH, VALIDATION_PATH]:
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