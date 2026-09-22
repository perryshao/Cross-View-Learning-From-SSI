'''reading skeleton data from UWA3D dataset '''
# python -m pdb read_skeletons_uwa_cv.py --view_1 ActionsSkeleton/view_1/ --view_2 ActionsSkeleton/view_2/ --view_3 ActionsSkeleton/view_3/ --view_4 ActionsSkeleton/view_4/
import numpy as np
import argparse
import os
import h5py
from sklearn import preprocessing

## ---------------------------------------------------------------------------
## Edit `filepath` to point at your local copy of the UWA3D-II dataset; the
## --view_N arguments are resolved relative to it.
## ---------------------------------------------------------------------------
filepath = '/home/data/UWA3DII/'

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-V1", "--view_1", required=True,
	help="path to the view 1 skeletons")
ap.add_argument("-V2", "--view_2", required=True,
	help="path to the view 2 skeletons")
ap.add_argument("-V3", "--view_3", required=True, 
	help="path to the view 3 skeletons")
ap.add_argument("-V4", "--view_4", required=True, 
	help="path to the view 4 skeletons")
    
args = vars(ap.parse_args())
view1_path = filepath+ap.parse_args().view_1
view2_path = filepath+ap.parse_args().view_2
view3_path = filepath+ap.parse_args().view_3
view4_path = filepath+ap.parse_args().view_4
path = [view1_path,view2_path,view3_path,view4_path]
joint_num = 15
max_len = 100 ## padding length
train_labels = []
test_labels = []
train_data_cv = []
test_data_cv = []
frame_length = []


def pad_sequences(sequences,max_len):
    y = list()
    for seq in sequences:
        if seq.shape[0] > max_len:
            seq = np.delete(seq,np.s_[max_len:],axis = 0)
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
    
for view_num in range(4):
    seqs = os.listdir(path[view_num])
    for seq in seqs:
        print('the %s sequence in %d' % (seq,view_num+1))
        seq_mat = path[view_num]+seq
        if seq == 'a26_s01_e01_v01.mat' or seq == 'a25_s01_e01_v01.mat':
            continue
        try:
            with h5py.File(seq_mat, 'r') as file:
                ske_xyz = file['A'][:]    
        except Exception:
            print('no file %s found!'% seq_mat)
            continue
        if ske_xyz != []:
            if (view_num+1) == 3 or (view_num+1) == 4 :##protocol (1,2) vs (3,4)
                ske_xyz = ske_xyz.reshape(-1,ske_xyz.shape[1]//3,3) #(t,joint_num,3)
                # for j in range(ske_xyz.shape[1]):
                    # ske_xyz[:,j,:] = savgol_filter(ske_xyz[:,j,:], 5, 2,axis=0)
                # ske_xyz = ske_xyz.reshape(ske_xyz.shape[0],-1)
                    
                    # mpl.rcParams['legend.fontsize'] = 10
                    # fig = plt.figure()
                    # ax = fig.gca(projection='3d')
                    # ax.plot(curve[:,0], curve[:,1], curve[:,2], label='parametric curve')
                    # ax.legend()
                    # plt.show()
                    # ax.plot(ske_xyz[:,0], ske_xyz[:,1], ske_xyz[:,2], label='parametric curve')
                    # ax.legend()
                    # plt.show()
                    
                test_data_cv.append(ske_xyz)
                test_labels.append(int(seq[1:3]))
                frame_length.append(ske_xyz.shape[0])
            else:
                ske_xyz = ske_xyz.reshape(-1,ske_xyz.shape[1]//3,3) #(t,joint_num,3)
                # for j in range(ske_xyz.shape[1]):
                    # ske_xyz[:,j,:] = savgol_filter(ske_xyz[:,j,:], 5, 2,axis=0)
                # ske_xyz = ske_xyz.reshape(ske_xyz.shape[0],-1)                
                train_data_cv.append(ske_xyz)
                train_labels.append(int(seq[1:3]))
                # slice = ske_xyz.shape[0]//10
                # for s in range(slice):
                    # train_data_cv.append(ske_xyz[s:s-slice])
                    # train_labels.append(int(seq[1:3])) 
                frame_length.append(ske_xyz.shape[0])
                
print('training data len:',len(train_data_cv))
print('sequence len:',max_len)
print('testing data len:',len(test_data_cv))
print('sequence len:',max_len)

print('to build training categorial labels for 30 classes ')
y_label = np.reshape(np.array(train_labels),(np.array(train_labels).shape[0],1))
y_label = y_label -1 # from 0 to 29 for 30 classes.
## here using LabelBinarizer to help create label indicator matrix from a list of multi-class labels
## same with keras.utils.to_categorical function
lb = preprocessing.LabelBinarizer()
lb.fit(np.unique(y_label))
y_train = lb.transform(y_label)
## y_train = keras.utils.to_categorical(y_label, num_classes=30)
print('to build testing categorial labels for 30 classes ')
y_label = np.reshape(np.array(test_labels),(np.array(test_labels).shape[0],1))
y_label = y_label -1 # from 0 to 29 for 30 classes.
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

scale1_list1 = [1,2,3,6,9,12,15]
scale1_list2 = [1,2,3,5,6,8,9,11,12,14,15]
scale1_list3 = range(1,16)
# scale1_list1 = [1,2,6,2,9,2,3,12,3,15]
# scale1_list2 = [1,2,5,6,5,2,8,9,8,2,3,11,12,11,3,14,15]
# scale1_list3 = [1,2,4,5,6,5,4,2,7,8,9,8,7,2,3,10,11,12,11,10,3,13,14,15]


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
x_train = np.zeros([x_train_full.shape[0],x_train_full.shape[1],len(scale1_list1), 3], dtype='float32')
for n in range(len(x_train_full)):
    for s in range(len(scale1_list1)):
        x_train[n][:,s] = x_train_full[n][:,scale1_list1[s]-1]

# x_train1,x_train2 = build_x_repeat(x_train)  
      
file1 = h5py.File(filepath+'Train_Raw_cv1.h5','w')
file1.create_dataset('x_train', data = x_train)
file1.create_dataset('y_train', data = y_train)    
file1.close()


## to pad the zeros to the joints that are beyond the 11th joint. so we do +1 on len(scale1_list2) 
padding_num = 1
x_train = np.zeros([x_train_full.shape[0],x_train_full.shape[1],len(scale1_list2)+padding_num,3], dtype='float32')
for n in range(len(x_train_full)):
    for s in range(len(scale1_list2)):
        x_train[n][:,s] = x_train_full[n,:,scale1_list2[s]-1]
    x_train[n][:,s+1:s+padding_num+1] = np.zeros([padding_num,3],dtype='float32') ## padding the joints which are 12, with zeros.

# x_train1,x_train2 = build_x_repeat(x_train)
        
file2 = h5py.File(filepath+'Train_Raw_cv2.h5','w')
file2.create_dataset('x_train', data = x_train)
file2.create_dataset('y_train', data = y_train)    
file2.close()

## to pad the zeros to the joints that are beyond the 20th joint. so we do +10 on len(scale1_list3) 
padding_num = 10
x_train = np.zeros([x_train_full.shape[0],x_train_full.shape[1],len(scale1_list3)+padding_num,3], dtype='float32')
for n in range(len(x_train_full)):
    for s in range(len(scale1_list3)):
        x_train[n][:,s] = x_train_full[n,:,scale1_list3[s]-1]
    x_train[n][:,s+1:s+padding_num+1] = np.zeros([padding_num,3],dtype='float32') ## padding the joints which are 16-25, with zeros.

# x_train1,x_train2 = build_x_repeat(x_train)
        
file3 = h5py.File(filepath+'Train_Raw_cv3.h5','w')
file3.create_dataset('x_train', data = x_train)
file3.create_dataset('y_train', data = y_train)    
file3.close()

###------------------##### testing set 
x_test = np.zeros([x_test_full.shape[0],x_test_full.shape[1],len(scale1_list1),3], dtype='float32')
for n in range(len(x_test_full)):
    for s in range(len(scale1_list1)):
        x_test[n][:,s] = x_test_full[n][:,scale1_list1[s]-1]

# x_test1,x_test2 = build_x_repeat(x_test)
        
file1 = h5py.File(filepath+'Test_Raw_cv1.h5','w')
file1.create_dataset('x_test', data = x_test)
file1.create_dataset('y_test', data = y_test)    
file1.close()


## to pad the zeros to the joints that are beyond the 11th joint. so we do +1 on len(scale1_list2) 
padding_num = 1
x_test = np.zeros([x_test_full.shape[0],x_test_full.shape[1],len(scale1_list2)+padding_num,3], dtype='float32')
for n in range(len(x_test_full)):
    for s in range(len(scale1_list2)):
        x_test[n][:,s] = x_test_full[n,:,scale1_list2[s]-1]
    x_test[n][:,s+1:s+padding_num+1] = np.zeros([padding_num,3],dtype='float32') ## padding the joints which are 12, with zeros.

# x_test1,x_test2 = build_x_repeat(x_test)
        
file2 = h5py.File(filepath+'Test_Raw_cv2.h5','w')
file2.create_dataset('x_test', data = x_test)
file2.create_dataset('y_test', data = y_test)    
file2.close()

## to pad the zeros to the joints that are beyond the 20th joint. so we do +10 on len(scale1_list3) 
padding_num = 10
x_test = np.zeros([x_test_full.shape[0],x_test_full.shape[1],len(scale1_list3)+padding_num,3], dtype='float32')
for n in range(len(x_test_full)):
    for s in range(len(scale1_list3)):
        x_test[n][:,s] = x_test_full[n,:,scale1_list3[s]-1]
    x_test[n][:,s+1:s+padding_num+1] = np.zeros([padding_num,3],dtype='float32') ## padding the joints which are 16-25, with zeros.

# x_test1,x_test2 = build_x_repeat(x_test)
        
file3 = h5py.File(filepath+'Test_Raw_cv3.h5','w')
file3.create_dataset('x_test', data = x_test)
file3.create_dataset('y_test', data = y_test)    
file3.close()

file4 = h5py.File(filepath+'frame_length.h5','w')
file4.create_dataset('frame_length', data = np.array(frame_length))
file4.close()



