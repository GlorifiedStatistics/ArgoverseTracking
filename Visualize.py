from Constants import *
from Datasets import *

_MAP_INC_START = 100
_MAP_X_SIZE = 5000
_MAP_Y_SIZE = 4500
_MAP_NORM_MULT = 1
_MAP_SCALE = 4.5
_MIN_MAP_SIZE = 400
_MAX_MAP_SIZE = 1200


def draw_map(points, norms, pin, pout):
    """
    Draws the road and the first car in the list's path
    :return:
    """

    points += [_MAP_INC_START, 0]

    minx = int(min(points[:, 0])) - 1
    maxx = int(max(points[:, 0])) + 1
    miny = int(min(points[:, 1])) - 1
    maxy = int(max(points[:, 1])) + 1

    image = np.ones([_MAP_Y_SIZE, _MAP_X_SIZE, 3])

    _l = progressbar.progressbar(range(len(points))) if len(points) > 100_000 else range(len(points))

    for i in _l:
        p, n = points[i], norms[i]
        p2 = (p + _MAP_NORM_MULT * n)
        cv2.line(image, tuple(p), tuple(p2), (0, 0, 255), thickness=1)

    for c, arr in zip([(255, 0, 0), (0, 255, 0)], [pin, pout]):
        for car in arr:
            print(len(arr))
            if car[0][0] == 0:
                print("AA")
                break
            for p in car:
                p = (int(p[0]), int(p[1]))
                cv2.circle(image, p, 1, c)
                if p[0] < minx:
                    minx = int(p[0]) - 1
                elif p[0] > maxx:
                    maxx = int(p[0]) + 1
                if p[1] < miny:
                    miny = int(p[1]) - 1
                elif p[1] > maxy:
                    maxy = int(p[1]) + 1

    image_slice = image[miny:maxy, minx:maxx, :3]

    if image_slice.shape[1] < _MIN_MAP_SIZE:
        sf = _MIN_MAP_SIZE / image_slice.shape[1]
        image_slice = cv2.resize(image_slice, (int(_MIN_MAP_SIZE), int(image_slice.shape[0] * sf)))
    elif image_slice.shape[0] < _MIN_MAP_SIZE:
        sf = _MIN_MAP_SIZE / image_slice.shape[0]
        image_slice = cv2.resize(image_slice, (int(image_slice.shape[1] * sf), int(_MIN_MAP_SIZE)))
    elif image_slice.shape[1] > _MAX_MAP_SIZE:
        sf = _MAX_MAP_SIZE / image_slice.shape[1]
        image_slice = cv2.resize(image_slice, (int(_MAX_MAP_SIZE), int(image_slice.shape[0] * sf)))
    elif image_slice.shape[0] > _MAX_MAP_SIZE:
        sf = _MAX_MAP_SIZE / image_slice.shape[0]
        image_slice = cv2.resize(image_slice, (int(image_slice.shape[1] * sf), int(_MAX_MAP_SIZE)))

    cv2.imshow("Image", image_slice)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def iter_maps():
    """
    Iterates through maps and displays them for a couple seconds each
    """
    for file in os.listdir(TRAINING_PATH):
        with open(os.path.join(TRAINING_PATH, file), 'rb') as f:
            data = pickle.load(f)
            draw_map(data['lane'][:, :2], data['lane_norm'][:, :2], data['p_in'], data['p_out'])


def data_histogram(hist):
    """
    Makes a histogram of the given histogram data
    :param hist: the values, bins tuple of histogram data
    """
    v, bins = hist

    for i in range(len(bins) - 1):
        bins[i] = (bins[i] + bins[i + 1]) / 2

    bins = bins[:-1]

    plt.bar(bins, v)
    plt.show()


def data_cleaning_norm_functions():
    stats = load_stats()
    name = 'p_step'

    f1 = NormFuncs.linear
    f2 = NormFuncs.std_step
    f3 = NormFuncs.tanh

    ps = stats['all'][name]

    c, v = ps['hist_x']
    c2, v2 = ps['hist_y']

    a = np.zeros([len(v) - 1, 2])
    a[:, 0] = ((v[1:] + v[:-1]) / 2)
    a[:, 1] = ((v2[1:] + v2[:-1]) / 2)

    y1 = f1(a, ps)[:, 0][c > 0]
    y2 = f2(a, ps)[:, 0][c > 0]
    y3 = f3(a, ps)[:, 0][c > 0]

    x = a[:, 0][c > 0]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Norm Function Errors of "%s"' % name)

    _y1 = abs(y1 - x)
    _y2 = abs(y2 - x)
    _y3 = abs(y3 - x)
    _max = max([max(_y1), max(_y2), max(_y3)])

    ax1.scatter(x, _y1)
    ax1.scatter(x, _y2, color='r')
    ax1.scatter(x, _y3, color='g')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(0, 4)

    ax2.scatter(x, _y1, label=f1.__name__)
    ax2.scatter(x, _y2, color='r', label=f2.__name__)
    ax2.scatter(x, _y3, color='g', label=f3.__name__)
    ax2.set_xlim(-6, 6)
    ax2.set_ylim(0, 8)
    ax2.legend()

    ax3.scatter(x, _y1)
    ax3.scatter(x, _y2, color='r')
    ax3.scatter(x, _y3, color='g')
    ax3.set_ylim(0, _max + 2)

    ax1.set(xlabel='value', ylabel='error')
    ax2.set(xlabel='value', ylabel='error')
    ax3.set(xlabel='value', ylabel='error')

    plt.show()
