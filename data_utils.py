"""Loading and reshaping of the NTU RGB+D skeleton data.

Shared by the three NTU training scripts.  The `.mat` files consumed here are
produced by the Matlab code in `preprocessing/NTU/Matlab`.
"""

import h5py
import keras
import numpy as np


def pad_sequences(sequences, max_len):
    """Pad/subsample every sequence to exactly `max_len` frames.

    Longer sequences are sampled with replacement and sorted in time;
    shorter ones are padded by
    repeating their last frame.
    """
    y = list()
    for seq in sequences:
        if seq.shape[0] > max_len:
            # Sample frame indices with replacement, then sort them.
            sample_frames = np.sort(np.random.choice(seq.shape[0], max_len))
            seq = seq[sample_frames, :]
        if seq.shape[0] < max_len:
            padding_array = np.repeat(
                np.reshape(seq[-1, :], [1, -1]), max_len - seq.shape[0], axis=0
            )
            seq = np.concatenate((seq, padding_array), axis=0)
        y.append(seq)
    return np.array(y)


def padding_pose(x):
    """Copy the first pose into an all-zero second pose, in place.

    NTU samples contain one or two people; both pose slots must be populated.
    """
    assert len(x.shape) == 2
    joints_xyz = x.shape[1]
    for n in range(x.shape[0]):
        if all(x[n, joints_xyz // 2 :] == 0):
            x[n, joints_xyz // 2 :] = x[n, : joints_xyz // 2]
    return x


def _read_split(filepath, mat_name, key, scale, max_len, class_num):
    """Read one train/test split out of a Matlab v7.3 `.mat` file."""
    f = h5py.File(filepath + mat_name + scale + '.mat', 'r')
    data = [f[element] for element in f[key][0]]
    label = [f[element] for element in f[key][1]]

    print('data len:', len(data))
    print('sequence len:', max_len)

    for i, sample in enumerate(data):
        data[i] = padding_pose(np.transpose(sample))

    y_label = [int(y_sample[0][0]) for y_sample in label]
    y_label = np.reshape(np.array(y_label), (len(y_label), 1))
    y_label = y_label - 1  # from 0 to 59 for 60 classes.
    y = keras.utils.to_categorical(y_label, num_classes=class_num)

    f.close()
    return data, y


def load_data(
    filepath,
    max_len,
    scale,
    train_mat='train_data_cv_scale',
    train_key='train_data_cv',
    test_mat='test_data_cv_scale',
    test_key='test_data_cv',
    class_num=60,
):
    '''Load and transform the raw skeleton data from *.mat files created by Matlab.

    The `*_mat` / `*_key` arguments exist because the late-fusion script reads
    the cross-view splits under their protocol names (`data_cv3` for training,
    `data_cv1` for testing) while the early-fusion scripts use generic names.
    '''
    # First, load the training data
    train_data, y_train = _read_split(filepath, train_mat, train_key, scale, max_len, class_num)

    # Second, load the testing data
    test_data, y_test = _read_split(filepath, test_mat, test_key, scale, max_len, class_num)

    y_train = y_train.astype('int8')
    y_test = y_test.astype('int8')

    # Pad or subsample sequences to a common length.
    print('Pad sequences (samples x time)')
    x_train = pad_sequences(train_data, max_len)
    x_test = pad_sequences(test_data, max_len)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    return [x_train, y_train, x_test, y_test]


def morph_data(filepath, max_len, scale, **load_kwargs):
    '''Reshape the raw data to the shape of
    dim 0: the number of samples
    dim 1: the number of frames
    dim 2: the number of joints
    dim 3: 3 (x,y,z)

    Outputs are saved as *.h5 files.
    '''
    # One load only: the previous version called load_data() twice and threw
    # away half of each result, doubling the disk I/O and peak memory.
    x_train, y_train, x_test, y_test = load_data(filepath, max_len, scale, **load_kwargs)

    x_train = np.reshape(x_train, [x_train.shape[0], x_train.shape[1], x_train.shape[2] // 3, 3])
    with h5py.File(filepath + 'Train_Raw_cv' + scale + '.h5', 'w') as file1:
        file1.create_dataset('x_train', data=x_train)
        file1.create_dataset('y_train', data=y_train)
    del x_train, y_train

    x_test = np.reshape(x_test, [x_test.shape[0], x_test.shape[1], x_test.shape[2] // 3, 3])
    with h5py.File(filepath + 'Test_Raw_cv' + scale + '.h5', 'w') as file2:
        file2.create_dataset('x_test', data=x_test)
        file2.create_dataset('y_test', data=y_test)
    del x_test, y_test


def load_raw(filepath, scale):
    """Load one scale of the morphed `Train_Raw_cv*.h5` / `Test_Raw_cv*.h5` pair."""
    with h5py.File(filepath + 'Train_Raw_cv' + scale + '.h5', 'r') as file1:
        x_train = file1['x_train'][:]
        y_train = file1['y_train'][:]
    with h5py.File(filepath + 'Test_Raw_cv' + scale + '.h5', 'r') as file2:
        x_test = file2['x_test'][:]
        y_test = file2['y_test'][:]
    return x_train, y_train, x_test, y_test


def save_metric_matrix(weights, scale='3'):
    """Dump the learned Mahalanobis transform L for inspection/visualisation.

    `weights` is the kernel of the corresponding `MetricLayer`, e.g.
    `model.get_layer('metric_layer').get_weights()[0]`.
    """
    with h5py.File('Metric_Matrix' + scale + '.h5', 'w') as file1:
        file1.create_dataset('matrix', data=weights)
