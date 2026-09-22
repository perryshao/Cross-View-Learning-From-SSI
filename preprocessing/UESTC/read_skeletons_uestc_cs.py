'''reading skeleton data from HRI-UESTC dataset '''
# python -m pdb read_skeletons_uestc_cs.py -p mat_from_skeleton/
import numpy as np
import scipy.io as io
import argparse
import os
import h5py
import keras
from keras.preprocessing import sequence
from sklearn import preprocessing
from scipy.signal import savgol_filter
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# construct the argument parse and parse the arguments
filepath = '/home/data/HRI-UESTC/'
datapath = '/home/data/HRI-UESTC/CS/' ##protocol (1,3,5,7) vs (2)
# filepath = '/home/data/HRI-UESTC/CV1/even-odd' ##protocol (2,4,6,FV) vs (1,3,5,7)
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
	help="path to the skeleton mat data")
    
args = vars(ap.parse_args())
mat_path = filepath+ap.parse_args().path
joint_num = 25
max_len = 150 ## padding length, mean: 200, best:300
train_labels = []
test_labels = []
train_data_cv = []
test_data_cv = []
frame_length = []


def pad_sequences(sequences,max_len):
    y = list()
    for seq in sequences:
        if seq.shape[0] > max_len:
            sample_frames = np.sort(np.random.choice(seq.shape[0],max_len)) ## uniform sample the frames
            seq = seq[sample_frames,:,:]
            # seq = np.delete(seq,np.s_[max_len:],axis = 0)
        if seq.shape[0] < max_len:
            padding_array = np.repeat(np.reshape(seq[-1,:,:],[1,-1,3]), max_len-seq.shape[0], axis=0)
            seq = np.concatenate((seq,padding_array),axis=0) 
        y.append(seq)
    return np.array(y)

def build_x_repeat(x):
    ## formulate the joint data as a set of repeating vector to build SSM in deep models
    x_repeat1 = np.zeros([x.shape[0], x.shape[1], np.square((x.shape[-1]//3)), 3],dtype="float32")
    for sample in range(x.shape[0]):
        for t in range(x.shape[1]):
            x_temp = np.repeat(np.reshape(x[sample,t],(1,-1)),x.shape[-1]//3, axis= 0)
            x_repeat1[sample,t,:,:] = np.reshape(x_temp,(-1,3))
           
    x_repeat2 = np.zeros([x.shape[0], x.shape[1], np.square((x.shape[-1]//3)), 3],dtype="float32")
    for sample in range(x.shape[0]):
        for t in range(x.shape[1]):
            x_temp = np.reshape(x[sample,t],(-1,3))
            x_repeat2[sample,t,:,:] = np.repeat(x_temp, x.shape[-1]//3, axis = 0)
    
    return [x_repeat1, x_repeat2]
    

seqs = os.listdir(mat_path)
for seq in seqs:
    seq_name_list = seq.strip().split('_')
    if int(seq_name_list[2][1:]) in [1, 2, 6, 12, 13, 16, 21, 24, 28, 29, 30, 31, 33, 35, 39, 41, 42, 45, 47, 50, 52, 54, 
                                   55, 57, 59, 61, 63, 64, 67, 69, 70, 71, 73, 77, 81, 84, 86, 87, 88, 90, 91, 93, 96, 99, 102, 
                                   103, 104, 107, 108, 112, 113] and seq_name_list[1][1:] not in ['8']: ##protocol: cross-subject 
        print('processing the %sth sequence' % seq)
        seq_mat = mat_path+seq
        try:
            file = io.loadmat(seq_mat)
            ske_xyz = file['v']    
        except Exception:
            print('no file %s found!'% seq)
            continue
        if ske_xyz != []:
            ske_xyz = ske_xyz.reshape(-1,ske_xyz.shape[1]//3,3) #(t,joint_num,3) 
            train_data_cv.append(ske_xyz)
            train_labels.append(int(seq_name_list[0][1:]))
            frame_length.append(ske_xyz.shape[0])   
            
    elif int(seq_name_list[2][1:]) not in [1, 2, 6, 12, 13, 16, 21, 24, 28, 29, 30, 31, 33, 35, 39, 41, 42, 45, 47, 50, 52, 54, 
                                   55, 57, 59, 61, 63, 64, 67, 69, 70, 71, 73, 77, 81, 84, 86, 87, 88, 90, 91, 93, 96, 99, 102, 
                                   103, 104, 107, 108, 112, 113] and seq_name_list[1][1:]  in ['7'] and seq_name_list[3][1:] == '2':  ##protocol: cross-subject
        print('processing the %sth sequence' % seq)
        seq_mat = mat_path+seq
        try:
            file = io.loadmat(seq_mat)
            ske_xyz = file['v']    
        except Exception:
            print('no file %s found!'% seq)
            continue
        if ske_xyz != []:
            ske_xyz = ske_xyz.reshape(-1,ske_xyz.shape[1]//3,3) #(t,joint_num,3)              
            test_data_cv.append(ske_xyz)
            test_labels.append(int(seq_name_list[0][1:]))
            frame_length.append(ske_xyz.shape[0])
                
print('training data len:',len(train_data_cv))
print('sequence len:',max_len)
print('testing data len:',len(test_data_cv))
print('sequence len:',max_len)

print('to build training categorial labels for 40 classes ')
y_label = np.reshape(np.array(train_labels),(np.array(train_labels).shape[0],1))
## here using LabelBinarizer to help create label indicator matrix from a list of multi-class labels
## same with keras.utils.to_categorical function
lb = preprocessing.LabelBinarizer()
lb.fit(np.unique(y_label))
y_train = lb.transform(y_label)
## y_train = keras.utils.to_categorical(y_label, num_classes=30)
print('to build testing categorial labels for 40 classes ')
y_label = np.reshape(np.array(test_labels),(np.array(test_labels).shape[0],1))
lb = preprocessing.LabelBinarizer()
lb.fit(np.unique(y_label))
y_test = lb.transform(y_label)
## y_test = keras.utils.to_categorical(y_label, num_classes=30)

print('Pad sequences (samples x time)')
x_train_full = pad_sequences(train_data_cv,max_len)
x_test_full = pad_sequences(test_data_cv,max_len)
# x_train_full = sequence.pad_sequences(train_data_cv,maxlen = max_len, dtype='float32', padding = 'post',truncating='post')
# x_test_full = sequence.pad_sequences(test_data_cv, maxlen = max_len, dtype='float32', padding = 'post',truncating='post')
print('x_train_full shape:', x_train_full.shape)
print('x_test_full shape:', x_test_full.shape)

scale_list1 = [4,21,1,20,16,24,22]
scale_list2 = [4,21,2,1,10,24,6,22,18,20,14,16]
scale_list3 = [4,3,21,2,1,9,10,11,25,12,24,5,6,7,23,8,22,17,18,19,20,13,14,15,16]

### eliminate those data which are zeros #######
x_train_has0 = x_train_full
sample_n = 0
for sample in range(x_train_has0.shape[0]):
    if np.count_nonzero(x_train_has0[sample]) == 0:
        continue
    x_train_full[sample_n] = x_train_has0[sample]
    sample_n += 1
x_train_full = np.delete(x_train_full,np.s_[sample_n:],axis=0)
y_train = np.delete(y_train,np.s_[sample_n:],axis=0)
 
x_test_has0 = x_test_full
sample_n = 0
for sample in range(x_test_has0.shape[0]):
    if np.count_nonzero(x_test_has0[sample]) == 0:
        continue
    x_test_full[sample_n] = x_test_has0[sample]
    sample_n += 1
x_test_full = np.delete(x_test_full,np.s_[sample_n:],axis=0)
y_test = np.delete(y_test,np.s_[sample_n:],axis=0)  
        
###------------------##### training set 
x_train = np.zeros([x_train_full.shape[0],x_train_full.shape[1],len(scale_list1), 3], dtype='float32')
for n in range(len(x_train_full)):
    for s in range(len(scale_list1)):
        x_train[n][:,s] = x_train_full[n][:,scale_list1[s]-1]
# x_train1,x_train2 = build_x_repeat(x_train)  
      
file1 = h5py.File(datapath+'Train_Raw_cv1.h5','w')
file1.create_dataset('x_train', data = x_train)
file1.create_dataset('y_train', data = y_train)    
file1.close()




x_train = np.zeros([x_train_full.shape[0],x_train_full.shape[1],len(scale_list2),3], dtype='float32')
for n in range(len(x_train_full)):
    for s in range(len(scale_list2)):
        x_train[n][:,s] = x_train_full[n,:,scale_list2[s]-1]
# x_train1,x_train2 = build_x_repeat(x_train)
        
file2 = h5py.File(datapath+'Train_Raw_cv2.h5','w')
file2.create_dataset('x_train', data = x_train)
file2.create_dataset('y_train', data = y_train)    
file2.close()


x_train = np.zeros([x_train_full.shape[0],x_train_full.shape[1],len(scale_list3),3], dtype='float32')
for n in range(len(x_train_full)):
    for s in range(len(scale_list3)):
        x_train[n][:,s] = x_train_full[n,:,scale_list3[s]-1]
# x_train1,x_train2 = build_x_repeat(x_train)
        
file3 = h5py.File(datapath+'Train_Raw_cv3.h5','w')
file3.create_dataset('x_train', data = x_train)
file3.create_dataset('y_train', data = y_train)    
file3.close()

###------------------##### testing set 
x_test = np.zeros([x_test_full.shape[0],x_test_full.shape[1],len(scale_list1),3], dtype='float32')
for n in range(len(x_test_full)):
    for s in range(len(scale_list1)):
        x_test[n][:,s] = x_test_full[n][:,scale_list1[s]-1]

# x_test1,x_test2 = build_x_repeat(x_test)
        
file1 = h5py.File(datapath+'Test_Raw_cv1.h5','w')
file1.create_dataset('x_test', data = x_test)
file1.create_dataset('y_test', data = y_test)    
file1.close()



x_test = np.zeros([x_test_full.shape[0],x_test_full.shape[1],len(scale_list2),3], dtype='float32')
for n in range(len(x_test_full)):
    for s in range(len(scale_list2)):
        x_test[n][:,s] = x_test_full[n,:,scale_list2[s]-1]
# x_test1,x_test2 = build_x_repeat(x_test)
        
file2 = h5py.File(datapath+'Test_Raw_cv2.h5','w')
file2.create_dataset('x_test', data = x_test)
file2.create_dataset('y_test', data = y_test)    
file2.close()


x_test = np.zeros([x_test_full.shape[0],x_test_full.shape[1],len(scale_list3),3], dtype='float32')
for n in range(len(x_test_full)):
    for s in range(len(scale_list3)):
        x_test[n][:,s] = x_test_full[n,:,scale_list3[s]-1]
# x_test1,x_test2 = build_x_repeat(x_test)
        
file3 = h5py.File(datapath+'Test_Raw_cv3.h5','w')
file3.create_dataset('x_test', data = x_test)
file3.create_dataset('y_test', data = y_test)    
file3.close()

file4 = h5py.File(filepath+'frame_length.h5','w')
file4.create_dataset('frame_length', data = np.array(frame_length))
file4.close()



