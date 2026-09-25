'''reading skeleton data from Northwestern-UCLA Multiview 3D dataset '''
# python -m pdb read_skeletons_ucla_cv.py --view_1 multiview_action/view_1 --view_2
# multiview_action/view_2 --view_3 multiview_action/view_3
import numpy as np
import argparse
import os
import h5py
from sklearn import preprocessing


def pad_sequences(sequences, max_len):
    if len(sequences) == 0:
        return np.empty((0, max_len, joint_num, 3), dtype='float32')
    y = list()
    for seq in sequences:
        if seq.shape[0] == 0:
            raise ValueError('Remove empty sequences and their labels before padding.')
        if seq.shape[0] > max_len:
            seq = np.delete(seq, np.s_[max_len:], axis=0)
        if seq.shape[0] < max_len:
            padding_array = np.repeat(
                np.reshape(seq[-1, :, :], [1, -1, 3]), max_len - seq.shape[0], axis=0
            )
            seq = np.concatenate((seq, padding_array), axis=0)
        y.append(seq)
    return np.array(y)


# Edit `filepath` to point at your local copy of the Northwestern-UCLA
# dataset; the --view_N arguments are resolved relative to it.
filepath = '/home/data/UCLA_Multiview3D/'

# Parse the command-line arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-V1", "--view_1", required=True, help="path to the view 1 skeletons")
ap.add_argument("-V2", "--view_2", required=True, help="path to the view 2 skeletons")
ap.add_argument("-V3", "--view_3", required=True, help="path to the view 3 skeletons")

args = vars(ap.parse_args())
view1_path = filepath + ap.parse_args().view_1
view2_path = filepath + ap.parse_args().view_2
view3_path = filepath + ap.parse_args().view_3
path = [view1_path, view2_path, view3_path]
joint_num = 20
max_len = 150  # padding length
train_labels = []
test_labels = []
train_data_cv = []
test_data_cv = []
frame_length = []
for view_num in range(3):
    seqs = os.listdir(path[view_num])
    for seq in seqs:
        print('the %s sequence in %d' % (seq, view_num + 1))
        seq_path = path[view_num] + '/' + seq
        files = os.listdir(seq_path)
        ske_slides = []
        ske_xyz = []
        try:
            with open(seq_path + '/' + files[files.index('fileList.txt')], 'r') as file:
                for line in file:
                    ske_slide = line.strip().split(' ')
                    ske_slides.append(
                        'frame_' + ske_slide[0] + '_tc_' + ske_slide[-1] + '_skeletons.txt'
                    )
        except Exception:
            print('no fileList.txt exist')
            continue
        for ske_slide in ske_slides:
            try:
                with open(seq_path + '/' + ske_slide, 'r') as file:
                    if not file.readline():
                        continue
                    for line in file:
                        if len(line.strip().split(',')) == 1:
                            break
                        ske_xyz.append(list(map(float, line.strip().split(',')[:-1])))
            except Exception:
                print('no file %s found!' % ske_slide)
                continue
        if np.size(ske_xyz) > 0:
            if (view_num + 1) == 3:  # protocol (1,2)-3
                test_data_cv.append(np.reshape(np.array(ske_xyz), (-1, joint_num, 3)))
                test_labels.append(int(seq[1:3]))
                frame_length.append(test_data_cv[-1].shape[0])
            else:
                train_data_cv.append(np.reshape(np.array(ske_xyz), (-1, joint_num, 3)))
                train_labels.append(int(seq[1:3]))
                frame_length.append(train_data_cv[-1].shape[0])

print('training data len:', len(train_data_cv))
print('sequence len:', max_len)
print('testing data len:', len(test_data_cv))
print('sequence len:', max_len)

print('to build training categorial labels for 10 classes ')
y_label = np.reshape(np.array(train_labels), (np.array(train_labels).shape[0], 1))
y_label = y_label - 1  # from 0 to 9 for 10 classes.
# here using LabelBinarizer to help create label indicator matrix from a list of multi-class labels
# Create a categorical label matrix.
lb = preprocessing.LabelBinarizer()
lb.fit(np.arange(10))  # Keep class columns identical across splits.
y_train = lb.transform(y_label) if y_label.size else np.empty((0, 10), dtype=int)
print('to build testing categorial labels for 10 classes ')
y_label = np.reshape(np.array(test_labels), (np.array(test_labels).shape[0], 1))
y_label = y_label - 1  # from 0 to 9 for 10 classes.
lb = preprocessing.LabelBinarizer()
lb.fit(np.arange(10))  # Keep class columns identical across splits.
y_test = lb.transform(y_label) if y_label.size else np.empty((0, 10), dtype=int)

print('Pad sequences (samples x time)')
x_train_full = pad_sequences(train_data_cv, max_len)
x_test_full = pad_sequences(test_data_cv, max_len)
print('x_train_full shape:', x_train_full.shape)
print('x_test_full shape:', x_test_full.shape)

scale1_list1 = [1, 3, 4, 8, 12, 16, 20]
scale1_list2 = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20]
scale1_list3 = range(1, 21)
# training set
x_train = np.zeros(
    [x_train_full.shape[0], x_train_full.shape[1], len(scale1_list1), 3], dtype='float32'
)
for n in range(len(x_train_full)):
    for s in range(len(scale1_list1)):
        x_train[n][:, s] = x_train_full[n][:, scale1_list1[s] - 1]

file1 = h5py.File(filepath + 'Train_Raw_cv1.h5', 'w')
file1.create_dataset('x_train', data=x_train)
file1.create_dataset('y_train', data=y_train)

file1.close()

x_train = np.zeros(
    [x_train_full.shape[0], x_train_full.shape[1], len(scale1_list2), 3], dtype='float32'
)
for n in range(len(x_train_full)):
    for s in range(len(scale1_list2)):
        x_train[n][:, s] = x_train_full[n][:, scale1_list2[s] - 1]

file2 = h5py.File(filepath + 'Train_Raw_cv2.h5', 'w')
file2.create_dataset('x_train', data=x_train)
file2.create_dataset('y_train', data=y_train)
file2.close()

# to pad the zeros to the joints that are beyond the 20th joint. so we do +5 on len(scale1_list3)
padding_num = 5
x_train = np.zeros(
    [x_train_full.shape[0], x_train_full.shape[1], len(scale1_list3) + padding_num, 3],
    dtype='float32',
)
for n in range(len(x_train_full)):
    for s in range(len(scale1_list3)):
        x_train[n][:, s] = x_train_full[n, :, scale1_list3[s] - 1]
    # padding the joints which are 21-25, with zeros.
    x_train[n][:, s + 1 : s + padding_num + 1] = np.zeros([padding_num, 3], dtype='float32')

file3 = h5py.File(filepath + 'Train_Raw_cv3.h5', 'w')
file3.create_dataset('x_train', data=x_train)
file3.create_dataset('y_train', data=y_train)
file3.close()

# testing set
x_test = np.zeros(
    [x_test_full.shape[0], x_test_full.shape[1], len(scale1_list1), 3], dtype='float32'
)
for n in range(len(x_test_full)):
    for s in range(len(scale1_list1)):
        x_test[n][:, s] = x_test_full[n][:, scale1_list1[s] - 1]

file1 = h5py.File(filepath + 'Test_Raw_cv1.h5', 'w')
file1.create_dataset('x_test', data=x_test)
file1.create_dataset('y_test', data=y_test)
file1.close()

x_test = np.zeros(
    [x_test_full.shape[0], x_test_full.shape[1], len(scale1_list2), 3], dtype='float32'
)
for n in range(len(x_test_full)):
    for s in range(len(scale1_list2)):
        x_test[n][:, s] = x_test_full[n][:, scale1_list2[s] - 1]

file2 = h5py.File(filepath + 'Test_Raw_cv2.h5', 'w')
file2.create_dataset('x_test', data=x_test)
file2.create_dataset('y_test', data=y_test)
file2.close()

# to pad the zeros to the joints that are beyond the 20th joint. so we do +5 on len(scale1_list3)
padding_num = 5
x_test = np.zeros(
    [x_test_full.shape[0], x_test_full.shape[1], len(scale1_list3) + padding_num, 3],
    dtype='float32',
)
for n in range(len(x_test_full)):
    for s in range(len(scale1_list3)):
        x_test[n][:, s] = x_test_full[n, :, scale1_list3[s] - 1]
    # padding the joints which are 21-25, with zeros.
    x_test[n][:, s + 1 : s + padding_num + 1] = np.zeros([padding_num, 3], dtype='float32')


file3 = h5py.File(filepath + 'Test_Raw_cv3.h5', 'w')
file3.create_dataset('x_test', data=x_test)
file3.create_dataset('y_test', data=y_test)
file3.close()

file4 = h5py.File(filepath + 'frame_length.h5', 'w')
file4.create_dataset('frame_length', data=np.array(frame_length))
file4.close()
