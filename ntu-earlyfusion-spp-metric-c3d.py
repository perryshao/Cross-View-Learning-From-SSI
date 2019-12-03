'''implement the ST-LSTM on NTU RGB+D dataset '''
## python -m pdb c3d-lstm-ntu-earlyfusion-v2-ta-metric.py
import time
import numpy as np
import matplotlib.pyplot as plt
import keras
import h5py
from keras.preprocessing import sequence
from keras.models import Model,Sequential
from keras.layers import Dense, LSTM, Permute,RepeatVector,GlobalAveragePooling2D, Dropout, Input, wrappers, Activation, MaxPooling3D, RNN, ActivityRegularization, Masking,Lambda,Conv3D,MaxPooling2D,GlobalMaxPooling2D,GRU,BatchNormalization
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from MetricLayer import MetricLayer
from keras.utils import plot_model
from keras import regularizers
from keras import backend as K
from keras.models import load_model
import keras.optimizers as op
import random
from skimage import io, data
import tensorflow as tf
from keras_spp.spp.SpatialPyramidPooling import SpatialPyramidPooling3D,SpatialPyramidPooling
from keras.models import model_from_json
from sklearn.externals import joblib
import colormaps as cmaps
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint,TensorBoard


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def visualize_layer(model, layer_name, data, time_step):
    layer_map = K.function([model.layers[0].input],[model_scale3.get_layer(layer_name).output])
    layer_output = layer_map([data])[0]
    f1 = layer_output[0,time_step,:,:,:]
    width = np.sqrt(f1.shape[-1])
    for channel in range(f1.shape[-1]):
        img_show = f1[:,:,channel]
        plt.subplot(width, width, channel+1)
        plt.subplot(width, width, channel+1)
        plt.imshow(img_show,cmap = cmaps.parula)
        plt.axis('off')
    plt.show()

def pad_sequences(sequences,max_len):
    y = list()
    for seq in sequences:
        if seq.shape[0] > max_len:
            sample_frames = np.sort(np.random.choice(seq.shape[0],max_len)) ## uniform sample the frames
            seq = seq[sample_frames,:]
            # seq = np.delete(seq,np.s_[max_len:],axis = 0)    
        if seq.shape[0] < max_len:
            padding_array = np.repeat(np.reshape(seq[-1,:],[1,-1]), max_len-seq.shape[0], axis=0)
            seq = np.concatenate((seq,padding_array),axis=0) 
        y.append(seq)
    return np.array(y)   

def padding_pose(x):
    assert len(x.shape) == 2
    joints_xyz = x.shape[1]
    for n in range(x.shape[0]):
        if all(x[n,joints_xyz//2:] == 0):
            x[n,joints_xyz//2:] = x[n,:joints_xyz//2]
    return x        

def load_data(filepath, max_len,scale):
    
    ###First, load the training data#########################################
    f1 = h5py.File(filepath+'train_data_cv_scale'+scale+'.mat', 'r')
    train_data = [f1[element] for element in f1['train_data_cv'][0]] 
    train_label = [f1[element] for element in f1['train_data_cv'][1]]

    
    print('training data len:',len(train_data))
    print('sequence len:',max_len)
    i = 0
    for train_sample in train_data:
      train_data[i] = np.transpose(train_sample)
      train_data[i] = padding_pose(train_data[i])
      i += 1
    y_label = []
    for y_sample in train_label:
       y_label.append(int(y_sample[0][0]))
    y_label = np.reshape(np.array(y_label),(np.array(y_label).shape[0],1))
    y_label = y_label -1 # from 0 to 59 for 60 classes.
    y_train = keras.utils.to_categorical(y_label, num_classes=60)
    
    ###First, load the testing data#########################################
    f2 = h5py.File(filepath+'test_data_cv_scale'+scale+'.mat', 'r')
    test_data = [f2[element] for element in f2['test_data_cv'][0]] 
    test_label = [f2[element] for element in f2['test_data_cv'][1]] 
    
    print('testing data len:',len(test_data))
    print('sequence len:',max_len)
    i = 0
    for test_sample in test_data:
      test_data[i] = np.transpose(test_sample)
      test_data[i] = padding_pose(test_data[i])
      i += 1
    y_label = []
    for y_sample in test_label:
       y_label.append(int(y_sample[0][0]))
    y_label = np.reshape(np.array(y_label),(np.array(y_label).shape[0],1))
    y_label = y_label -1 # from 0 to 59 for 60 classes.
    y_test = keras.utils.to_categorical(y_label, num_classes=60)

    y_test.astype('int8')
    y_train.astype('int8')
    f1.close()
    f2.close()

    ###########padding the sequences with various lengths#####################
    print('Pad sequences (samples x time)')
    x_train = pad_sequences(train_data,max_len)
    x_test = pad_sequences(test_data,max_len)
    # x_train = sequence.pad_sequences(train_data, maxlen = max_len, dtype = 'float32', padding = 'post',truncating='post')
    # x_test = sequence.pad_sequences(test_data, maxlen = max_len,  dtype = 'float32', padding = 'post',truncating='post')
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)   
   
    # file1 = h5py.File('Trainset.h5','w')
    # file1.create_dataset('X_train', data = x_train)
    # file1.create_dataset('Y_train', data = y_train)
    # file1.close()
    # file2 = h5py.File('Testset.h5','w')
    # file2.create_dataset('X_test', data = x_test)
    # file2.create_dataset('Y_test', data = y_test)
    # file2.close()
    # reshape x_train x_test to 5 dimenstion tensor for convLSTM
    return [x_train, y_train, x_test, y_test]
    
def morph_data(filepath,max_len,scale):
    x_train, y_train, _, _ = load_data(filepath,max_len,scale)
    x_train = np.reshape(x_train,[x_train.shape[0],x_train.shape[1],x_train.shape[2]//3, 3])
    file1 = h5py.File(filepath+'Train_Raw_cv'+scale+'.h5','w')
    file1.create_dataset('x_train', data = x_train)
    file1.create_dataset('y_train', data = y_train)
    file1.close()
    del x_train
    del y_train
    
    _, _, x_test, y_test = load_data(filepath,max_len,scale)
    x_test = np.reshape(x_test,[x_test.shape[0],x_test.shape[1],x_test.shape[2]//3, 3])
    file2 = h5py.File(filepath+'Test_Raw_cv'+scale+'.h5','w')
    file2.create_dataset('x_test', data = x_test)
    file2.create_dataset('y_test', data = y_test)
    file2.close()
    
def generate_train(filepath,batch_size,scale):
    while 1:
        file1 = h5py.File(filepath+'Train_Raw_cv'+scale+'.h5','r')
        samples = file1['x_train'].shape[0]
        if samples%batch_size == 0:
            batch_times = (samples/batch_size)
        else:
            batch_times = (samples/batch_size)+1
        for cnt in range(batch_times-1): 
            X_train = file1['x_train'][cnt*batch_size:(cnt+1)*batch_size]
            Y_train = file1['y_train'][cnt*batch_size:(cnt+1)*batch_size]
            yield (X_train,Y_train)
        cnt = cnt+1
        X_train = file1['x_train'][cnt*batch_size:]
        Y_train = file1['y_train'][cnt*batch_size:]
        x = X_train
        y = Y_train
        yield(x,y)
    file1.close()

def generate_test(filepath, batch_size,scale):
    while 1:
        file1 = h5py.File(filepath+'Test_Raw_cv'+scale+'.h5','r')
        samples = file1['x_test'].shape[0]
        if samples%batch_size == 0:
            batch_times = (samples/batch_size)
        else:
            batch_times = (samples/batch_size)+1
        for cnt in range(batch_times-1): 
            X_test = file1['x_test'][cnt*batch_size:(cnt+1)*batch_size]
            Y_test = file1['y_test'][cnt*batch_size:(cnt+1)*batch_size]
            yield(X_test,Y_test)
        cnt = cnt+1
        X_test = file1['x_test'][cnt*batch_size:]
        Y_test = file1['y_test'][cnt*batch_size:]
        x = X_test
        y = Y_test
        yield(x,y)
    file1.close()
        
        # file2 = h5py.File('Testset.h5','r')
        # X_test = file2['X_test'][:]
        # Y_test = file2['Y_test'][:]
        
        # file1.close()
        # file2.close()
        # for x, y in train_list:
            # yield(x,y)
def generate_train_fusion(filepath,batch_size):
    while 1:
        file1 = h5py.File(filepath+'Train_Raw_cv1'+'.h5','r')
        file2 = h5py.File(filepath+'Train_Raw_cv2'+'.h5','r')
        file3 = h5py.File(filepath+'Train_Raw_cv3'+'.h5','r')
        samples = file1['x_train'].shape[0]
        if samples%batch_size == 0:
            batch_times = (samples/batch_size)
        else:
            batch_times = (samples/batch_size)+1
        for cnt in range(batch_times-1): 
            X_train1 = file1['x_train'][cnt*batch_size:(cnt+1)*batch_size]
            Y_train1 = file1['y_train'][cnt*batch_size:(cnt+1)*batch_size]
            X_train2 = file2['x_train'][cnt*batch_size:(cnt+1)*batch_size]
            Y_train2 = file2['y_train'][cnt*batch_size:(cnt+1)*batch_size]
            X_train3 = file3['x_train'][cnt*batch_size:(cnt+1)*batch_size]
            Y_train3 = file3['y_train'][cnt*batch_size:(cnt+1)*batch_size]
            x = [X_train1,X_train2,X_train3]
            y = Y_train1
            yield (x,y)
            
        cnt = cnt+1
        X_train1 = file1['x_train'][cnt*batch_size:]
        Y_train1 = file1['y_train'][cnt*batch_size:]
        X_train2 = file2['x_train'][cnt*batch_size:]
        Y_train2 = file2['y_train'][cnt*batch_size:]
        X_train3 = file3['x_train'][cnt*batch_size:]
        Y_train3 = file3['y_train'][cnt*batch_size:]
        x = [X_train1,X_train2,X_train3]
        y = Y_train1
        yield(x,y)
    file1.close()
    file2.close()
    file3.close()
    
def generate_test_fusion(filepath, batch_size):
    while 1:
        file1 = h5py.File(filepath+'Test_Raw_cv1'+'.h5','r')
        file2 = h5py.File(filepath+'Test_Raw_cv2'+'.h5','r')
        file3 = h5py.File(filepath+'Test_Raw_cv3'+'.h5','r')
        samples = file1['x_test'].shape[0]
        if samples%batch_size == 0:
            batch_times = (samples/batch_size)
        else:
            batch_times = (samples/batch_size)+1
        for cnt in range(batch_times-1): 
            X_test1 = file1['x_test'][cnt*batch_size:(cnt+1)*batch_size]
            Y_test1 = file1['y_test'][cnt*batch_size:(cnt+1)*batch_size]
            X_test2 = file2['x_test'][cnt*batch_size:(cnt+1)*batch_size]
            Y_test2 = file2['y_test'][cnt*batch_size:(cnt+1)*batch_size]
            X_test3 = file3['x_test'][cnt*batch_size:(cnt+1)*batch_size]
            Y_test3 = file3['y_test'][cnt*batch_size:(cnt+1)*batch_size]
            x = [X_test1,X_test2,X_test3]
            y = Y_test1
            yield (x,y)
            
        cnt = cnt+1
        X_test1 = file1['x_test'][cnt*batch_size:]
        Y_test1 = file1['y_test'][cnt*batch_size:]
        X_test2 = file2['x_test'][cnt*batch_size:]
        Y_test2 = file2['y_test'][cnt*batch_size:]
        X_test3 = file3['x_test'][cnt*batch_size:]
        Y_test3 = file3['y_test'][cnt*batch_size:]
        x = [X_test1,X_test2,X_test3]
        y = Y_test1
        yield(x,y)
    file1.close()
    file2.close()
    file3.close()
    
def distribute_alpha_t(x):
    return K.repeat_elements(x,3,axis=2)
def dis_alpha_t_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3  # only valid for 3D tensors
    shape= [input_shape[0],input_shape[1],input_shape[2]*3]
    return tuple(shape)

def flattenConv(x):
    # shape = x.shape.as_list()
    # dim = np.prod(shape[2:])
    shape = K.shape(x)
    dim = K.prod(shape[2:])
    return K.reshape(x,[shape[0],shape[1],dim])
def flattenConv_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4 or len(shape) == 5 # only valid for 4D or 5D tensors
    dim = np.prod(shape[2:])
    shape= [input_shape[0],input_shape[1],dim]
    return tuple(shape)  

def flatten3dConv(x):
    # shape = x.shape.as_list()
    # dim = np.prod(shape[2:])
    shape = K.shape(x)
    dim = K.prod(shape[1:])/8
    return K.reshape(x,[shape[0],shape[1]/4,dim])
def flatten3dConv_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 5  # only valid for 5D tensors
    shape= [input_shape[0],shape[1]/4,4*input_shape[2]*input_shape[3]*input_shape[4]]
    return tuple(shape)
    
def unflattenConv(x):
    shape = K.shape(x)
    # shape = x.shape.as_list()
    return K.reshape(x,[-1,shape[1],K.cast(K.sqrt(shape[2]),dtype='int32'),K.cast(K.sqrt(shape[2]),dtype='int32'),1])
def unflattenConv_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3  # only valid for 3D tensors
    shape= [input_shape[0],input_shape[1],K.cast(K.sqrt(input_shape[2]),dtype='int32'),K.cast(K.sqrt(input_shape[2]),dtype='int32'),1]
    return tuple(shape)

def flattenSSM(x):
    shape = x.shape.as_list()
    # dim = np.prod(shape[2:])
    # shape = K.shape(x)
    return K.reshape(x,[-1,shape[1], int(np.sqrt(shape[2])),int(np.sqrt(shape[2])), 3])
def flattenSSM_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4  # only valid for 4D tensors
    shape= [input_shape[0],input_shape[1],int(np.sqrt(input_shape[2])),int(np.sqrt(input_shape[2])),3]
    return tuple(shape)
def repeat_x_onejoint(x):
    shape = x.shape.as_list()
    x = K.repeat_elements(x, shape[-2], axis = -2)
    return x
def repeat_x_onejoint_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4 # only valid for 4D tensors
    shape = [input_shape[0],input_shape[1],input_shape[2]*input_shape[2],input_shape[-1]]
    return tuple(shape)
def repeat_x_groupjoint(x):
    shape = x.shape.as_list()
    x = K.repeat_elements(x, shape[-2], axis = -3)
    x = K.reshape(x,[-1,shape[1],shape[-2]*shape[-2],shape[-1]])
    return x
def repeat_x_groupjoint_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4 # only valid for 4D tensors
    shape = [input_shape[0],input_shape[1],input_shape[2]*input_shape[2],input_shape[-1]]
    return tuple(shape)

    # compute ssm using euclidean distance
def compute_ssm(x):
    # x = K.sqrt(K.dot(K.square(x), K.ones(shape=(3, 1))))
    x = K.sqrt(K.sum(K.square(x), axis = -1, keepdims=True))
    x_st = K.reshape(x,[-1,x.shape[1]*x.shape[2],x.shape[-1]])
    x_st_max = K.reshape(K.max(x_st,axis=-2,keepdims=True),[-1,1,1,x.shape[-1]])
    x_st_min = K.reshape(K.min(x_st,axis=-2,keepdims=True),[-1,1,1,x.shape[-1]])
    x = (x - x_st_min)/(x_st_max - x_st_min+K.epsilon())
    x = K.repeat_elements(x,3, axis= -1)   # compose 3 channels for c3d pretrained net 
    # x = (x - K.min(x,axis=2,keepdims=True))/(K.max(ssm,axis=2,keepdims=True) - K.min(ssm,axis=2,keepdims=True)+K.epsilon())
    # x_std = K.std(x, axis=2, keepdims=True)
    # x = (x-K.repeat_elements(K.mean(x,axis=2,keepdims=True),x.shape[-2],axis=2))/(K.repeat_elements(x_std,x.shape[-2],axis=2)+K.epsilon())
    return x

def compute_ssm_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4 # only valid for 4D tensors
    shape = [input_shape[0], input_shape[1], input_shape[2], 3] # compose 3 channels for c3d pretrained net 
    return shape   

def save_metric_matrix(weights, scale ='3'):
    file1 = h5py.File('Metric_Matrix'+scale+'.h5','w')
    file1.create_dataset('matrix', data = weights)
    file1.close()

class LrReducer(keras.callbacks.Callback):
    def __init__(self, patience=5, reduce_rate=0.1, reduce_nb=10, verbose=1):
        super(LrReducer, self).__init__()
        self.patience = patience
        self.wait = 0
        self.best_loss = 100.
        self.reduce_rate = reduce_rate
        self.current_reduce_nb = 0
        self.reduce_nb = reduce_nb
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current_loss = logs.get('val_loss')
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
            if self.verbose > 0:
                print('---current best val loss: %.3f' % current_loss)
        else:
            if self.wait >= self.patience:
                self.current_reduce_nb += 1
                if self.current_reduce_nb <= self.reduce_nb:
                    lr = K.get_value(self.model.optimizer.lr)
                    K.set_value(self.model.optimizer.lr,lr*self.reduce_rate)
                    print("reduce lr by dividing 10x")
                else:
                    if self.verbose > 0:
                        print("Epoch %d: early stopping" % (epoch))
                    self.model.stop_training = True
            else:
                if self.verbose > 0:
                    print("current loss > best_loss, but doesn't reach the patience epochs")
            self.wait += 1
            
def get_model(summary=False):
    """ Return the Keras model of the network
    """
    model = Sequential()
    # 1st layer group
    model.add(Convolution3D(64, [3, 3, 3], activation='relu', 
                            padding='same', name='conv1',
                            strides=(1, 1, 1), 
                            input_shape=(16, 112, 112, 3)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), 
                           padding='valid', name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(128, [3, 3, 3], activation='relu', 
                            padding='same', name='conv2',
                            strides=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           padding='valid', name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(256, [3, 3, 3], activation='relu', 
                            padding='same', name='conv3a',
                            strides=(1, 1, 1)))
    model.add(Convolution3D(256, [3, 3, 3], activation='relu', 
                            padding='same', name='conv3b',
                            strides=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           padding='valid', name='pool3'))
    # 4th layer group
    model.add(Convolution3D(512, [3, 3, 3], activation='relu', 
                            padding='same', name='conv4a',
                            strides=(1, 1, 1)))
    model.add(Convolution3D(512, [3, 3, 3], activation='relu', 
                            padding='same', name='conv4b',
                            strides=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           padding='valid', name='pool4'))
    # 5th layer group
    model.add(Convolution3D(512, [3, 3, 3], activation='relu', 
                            padding='same', name='conv5a',
                            strides=(1, 1, 1)))
    model.add(Convolution3D(512, [3, 3, 3], activation='relu', 
                            padding='same', name='conv5b',
                            strides=(1, 1, 1)))
    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           padding='valid', name='pool5'))
    model.add(Flatten())
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5))
    model.add(Dense(487, activation='softmax', name='fc8'))
    if summary:
        print(model.summary())
    return model
    
def get_int_model(layer, backend='tf',InputShape=None):

    if backend == 'tf':
        input_shape=(16, 112, 112, 3) # l, h, w, c
    else:
        input_shape=(3, 16, 112, 112) # c, l, h, w
    if InputShape != None:
        input_shape = InputShape # l, h, w, c

    int_model = Sequential()

    int_model.add(Convolution3D(64, [3, 3, 3], activation='relu',
                            padding='same', name='conv1',
                            input_shape=input_shape))
    if layer == 'conv1':
        return int_model
    int_model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           padding='valid', name='pool1'))
    if layer == 'pool1':
        return int_model

    # 2nd layer group
    int_model.add(Convolution3D(128, [3, 3, 3], activation='relu',
                            padding='same', name='conv2'))
    if layer == 'conv2':
        return int_model
    int_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool2'))
    if layer == 'pool2':
        return int_model

    # 3rd layer group
    int_model.add(Convolution3D(256, [3, 3, 3], activation='relu',
                            padding='same', name='conv3a'))
    if layer == 'conv3a':
        return int_model
    int_model.add(Convolution3D(256, [3, 3, 3], activation='relu',
                            padding='same', name='conv3b'))
    if layer == 'conv3b':
        return int_model
    int_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool3'))
    if layer == 'pool3':
        return int_model

    # 4th layer group
    int_model.add(Convolution3D(512, [3, 3, 3], activation='relu',
                            padding='same', name='conv4a'))
    if layer == 'conv4a':
        return int_model
    int_model.add(Convolution3D(512, [3, 3, 3], activation='relu',
                            padding='same', name='conv4b'))
    if layer == 'conv4b':
        return int_model
    int_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool4'))
    if layer == 'pool4':
        return int_model

    # 5th layer group
    int_model.add(Convolution3D(512, [3, 3, 3], activation='relu',
                            padding='same', name='conv5a'))
    if layer == 'conv5a':
        return int_model
    int_model.add(Convolution3D(512, [3, 3, 3], activation='relu',
                            padding='same', name='conv5b'))
    if layer == 'conv5b':
        return int_model
    int_model.add(ZeroPadding3D(padding=(0, 1, 1), name='zeropad'))
    int_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool5'))
    if layer == 'pool5':
        return int_model

    int_model.add(Flatten())
    # FC layers group
    int_model.add(Dense(4096, activation='relu', name='fc6'))
    if layer == 'fc6':
        return int_model
    int_model.add(Dropout(.5))
    int_model.add(Dense(4096, activation='relu', name='fc7'))
    if layer == 'fc7':
        return int_model
    int_model.add(Dropout(.5))
    int_model.add(Dense(487, activation='softmax', name='fc8'))
    if layer == 'fc8':
        return int_model

    return None

def train_lstms(layers_num,batch_size,joint_num,Regs,filepath,stride_set,spp_numbers,epoch,fine_tune = False,scale='1'):
    global_start_time = time.time()
    max_len = 150
    class_num = 60
    results=[]
    R1 = Regs[0]
    R2 = Regs[1]
    R3 = Regs[2]
    clip_length = 35
    
    print('> Loading data... ')
    
    file1 = h5py.File(filepath+'Train_Raw_cv'+scale+'.h5','r')
    file2 = h5py.File(filepath+'Test_Raw_cv'+scale+'.h5','r')
    x_train = file1['x_train'][:]
    y_train = file1['y_train'][:]
    x_test = file2['x_test'][:]
    y_test = file2['y_test'][:]

    print('x_train shape:', x_train.shape)  #(40091L, 150L,rows,cols,channels)
    print('y_train shape:', y_train.shape)  #(40091L,60L)
    print('x_test shape:',  x_test.shape)   #(16487L, 150L,rows,cols,channels)
    print('y_test shape',   y_test.shape)   #(16487L,60L)
    
    file1.close()
    file2.close()
    print('> Data Loaded. Compiling...')
    
    ################define the model############################
    model = get_model(layers_num,joint_num, max_len, stride_set,class_num, R3,spp_numbers,scale)
    if fine_tune == True:
        model.load_weights('model_scale1_v2_ta.h5')
        # for l in model.layers[:-6]: # freeze the convlution layers, tune the lstm layers
            # l.trainable = False
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    model.summary()
    plot_model(model,to_file='model_scale'+scale+'.png')
    ## using multiple GPU model
    mgpu_model = keras.utils.multi_gpu_model(model,gpus=2)
    mgpu_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    mgpu_model.summary()
    start_time = time.time()
    
    print('beginning to train the model....')
    
    history = mgpu_model.fit(x_train,y_train, batch_size= batch_size, epochs=epoch)


    average_time_per_epoch = (time.time() - start_time) / epoch
    results.append((history, average_time_per_epoch))
    print('Training duration (s) : ', time.time() - global_start_time)
    ################evaluate the model by using the test set##############################
    scores =mgpu_model.evaluate(x_test,y_test, batch_size= batch_size)
    print('Test score:', scores[0])
    print('Test accuracy:', scores[1])

    # predicted = model.predict([X_test,X_test,X_test_1])
    # print('predicted shape:',np.array(predicted).shape)  #(16488L,1L)
    # predicted = np.reshape(predicted, (predicted.size,)) #(16488L,)

    print('Training duration (s) : ', time.time() - global_start_time)
    
    plot_model(model,to_file='model.png')
    
    return [model,scores[1]]

if __name__=='__main__':
    global_start_time = time.time()
    epochs = [60,60]
    max_len = 150
    class_num = 60
    lamda1 = 0.01
    lamda2 = 0.001
    # lamda3 = 0.00001
    # lamda3 = 0.0000001
    lamda3 = 0.000001
    test_score=[0,0,0]
    results = []
    stride_set = [1,1,1]
    # R1 = regularizers.alpha_reg(lamda1,max_len)
    R1 = 0
    R2 = lamda2/max_len
    R3 = regularizers.l2(lamda3)
    Regs = [R1,R2,R3]
    clip_length = 35
    filepath = '/home/data/nturgbd_skeletons/ntu_data_mat/'
    
    # print('morphing the data into [samples, t, joint_num,3] format....')
    # morph_data(filepath, max_len,scale ='1') 
    # morph_data(filepath, max_len,scale ='2') 
    # morph_data(filepath, max_len,scale ='3') 
    
    ###########################################################
    joint_num1  = 7*2 
    spp_numbers1 = [0,1,2] # sum([ x**2 for x in spp_numbers1 ])*16 for the spp layer
    layers_num1 = [joint_num1,4,8,16,50,2*2*16] ## 16,32,64,100##64,64,64,200 [input_layer,lstm1-lstm3,fc1_layer,fc2_layer] [joint_num1,50,50,50,100]   
    #########################################################
    joint_num2  = 12*2
    spp_numbers2 = [0,1,2]
    layers_num2 = [joint_num2,8,16,32,50,2*2*32] ##128, 32,32,100,100##64,64,64,200 [input_layer,lstm1-lstm3,fc1_layer,fc2_layer] [joint_num2,80,80,80,160]
    ##########################################################
    joint_num3  = 25*2
    spp_numbers3 = [1,2,3]
    layers_num3 = [joint_num3,16,32,64,50,4*4*64] ##576, 32,64,64,200 [input_layer,lstm1-lstm3,fc1_layer,fc2_layer] [joint_num2,100,100,100,200]
    ###########################################################
    lstm_layer_num = 100
    fc1_layer = 50
    print("[Info] Loading the Convolutional Layers for the First Stream")
    model_scale1 = get_int_model('pool3', backend='tf', InputShape=(max_len, joint_num1, joint_num1, 3))
    model_scale1.summary()
    print("[Info] Loading the Convolutional Layers -- DONE!")
    print("[Info] Loading the Convolutional Layers for the Second Stream")
    model_scale2 = get_int_model('conv4b', backend='tf', InputShape=(max_len, joint_num2, joint_num2, 3))
    model_scale2.summary()
    print("[Info] Loading the Convolutional Layers -- DONE!")
    print("[Info] Loading the Convolutional Layers for the Third Stream")
    model_scale3 = get_int_model('conv4b', backend='tf', InputShape=(max_len, joint_num3, joint_num3, 3))
    model_scale3.summary()
    print("[Info] Loading the Convolutional Layers -- DONE!")
    model_scale1.name ='model_1'
    model_scale2.name ='model_2'
    model_scale3.name ='model_3' # to avoid name conflicts of different models
    model_scale1.load_weights('sports1M_weights_tf.h5',by_name=True)
    model_scale2.load_weights('sports1M_weights_tf.h5',by_name=True)
    model_scale3.load_weights('sports1M_weights_tf.h5',by_name=True)
    # model_scale1.load_weights('model_scale1_v3_ta.h5',by_name=True)
    # model_scale2.load_weights('model_scale2_v3_ta.h5',by_name=True)
    # model_scale3.load_weights('model_scale3_v3_ta.h5',by_name=True)
    
    #####define the inputs
    main_input1= Input(shape=(max_len,joint_num1,3),dtype = 'float32')
    main_input2= Input(shape=(max_len,joint_num2,3),dtype = 'float32')	
    main_input3= Input(shape=(max_len,joint_num3,3),dtype = 'float32')
     ## define the input shape
    repeat_input1_1 = Lambda(repeat_x_onejoint,repeat_x_onejoint_output_shape)(main_input1)
    repeat_input1_2 = Lambda(repeat_x_groupjoint,repeat_x_groupjoint_output_shape)(main_input1)
    repeat_input2_1 = Lambda(repeat_x_onejoint,repeat_x_onejoint_output_shape)(main_input2)
    repeat_input2_2 = Lambda(repeat_x_groupjoint,repeat_x_groupjoint_output_shape)(main_input2)
    repeat_input3_1 = Lambda(repeat_x_onejoint,repeat_x_onejoint_output_shape)(main_input3)
    repeat_input3_2 = Lambda(repeat_x_groupjoint,repeat_x_groupjoint_output_shape)(main_input3)
    ########compute the ssm input###########
    ssm_input1 = keras.layers.subtract([repeat_input1_1,repeat_input1_2])
    ssm_input2 = keras.layers.subtract([repeat_input2_1,repeat_input2_2])
    ssm_input3 = keras.layers.subtract([repeat_input3_1,repeat_input3_2])
    # ssm_input = wrappers.TimeDistributed(MetricLayer(joint_num))(ssm_input)
    # ssm_input = MetricLayer(joint_num,kernel_regularizer = regularizers.Maha_R(lamda4))(ssm_input)
    ssm_input1 = Lambda(compute_ssm,compute_ssm_output_shape)(ssm_input1)
    ssm_input1 = Lambda(flattenSSM,flattenSSM_output_shape)(ssm_input1)
    ssm_input2 = Lambda(compute_ssm,compute_ssm_output_shape)(ssm_input2)
    ssm_input2 = Lambda(flattenSSM,flattenSSM_output_shape)(ssm_input2)
    ssm_input3 = Lambda(compute_ssm,compute_ssm_output_shape)(ssm_input3)
    ssm_input3 = Lambda(flattenSSM,flattenSSM_output_shape)(ssm_input3)
    
    ####determine the number of batches#########################
    batch_size = 16
    file1 = h5py.File(filepath+'Train_Raw_cv1.h5','r')
    file2 = h5py.File(filepath+'Test_Raw_cv1.h5','r')
    x_train1 = file1['x_train'][:]
    y_train1 = file1['y_train'][:]
    x_test1 = file2['x_test'][:]
    y_test1 = file2['y_test'][:]
    file1.close()
    file2.close()

    file1 = h5py.File(filepath+'Train_Raw_cv2.h5','r')
    file2 = h5py.File(filepath+'Test_Raw_cv2.h5','r')
    x_train2 = file1['x_train'][:]
    y_train2 = file1['y_train'][:]
    x_test2 = file2['x_test'][:]
    y_test2 = file2['y_test'][:]
    file1.close()
    file2.close()

    file1 = h5py.File(filepath+'Train_Raw_cv3.h5','r')
    file2 = h5py.File(filepath+'Test_Raw_cv3.h5','r')
    x_train3 = file1['x_train'][:]
    y_train3 = file1['y_train'][:]
    x_test3 = file2['x_test'][:]
    y_test3 = file2['y_test'][:]
    file1.close()
    file2.close()

    print('x_train shape:', x_train1.shape,x_train2.shape,x_train3.shape)  #(40091L, 150L,rows,cols,channels)
    print('y_train shape:', y_train1.shape,y_train2.shape,y_train3.shape)  #(40091L,60L)
    print('x_test shape:',  x_test1.shape,x_test2.shape,x_test3.shape)   #(16487L, 150L,rows,cols,channels)
    print('y_test shape',   y_test1.shape,y_test2.shape,y_test3.shape)   #(16487L,60L)

    # visualize_layer(model_scale3, 'conv3d_15', x_train3[5:6], 35)

    if 1: # feature fusion
        
        ######### input to the three models
        c3d_out1 = model_scale1(ssm_input1)
        c3d_out2 = model_scale2(ssm_input2)
        c3d_out3 = model_scale3(ssm_input3)
        
        # fc_output1 = BatchNormalization(name='BN_layer1')(fc_output1)
        # fc_output2 = BatchNormalization(name='BN_layer2')(fc_output2)
        # fc_output3 = BatchNormalization(name='BN_layer3')(fc_output3)
        feat_output1 = Lambda(flattenConv,flattenConv_output_shape)(c3d_out1)
        feat_output2 = Lambda(flattenConv,flattenConv_output_shape)(c3d_out2) 
        feat_output3 = Lambda(flattenConv,flattenConv_output_shape)(c3d_out3)      
        feat_combined = keras.layers.concatenate([feat_output1,feat_output2,feat_output3])       
        feat_combined_bn = BatchNormalization(name='BN_layer')(feat_combined)
        lstm_output1 = LSTM(lstm_layer_num,return_sequences=True, dropout=0.5,recurrent_regularizer = R3,kernel_regularizer = R3,bias_regularizer = R3,
        name = 'lstm1')(feat_combined_bn)
        lstm_output2 = LSTM(lstm_layer_num,return_sequences=True, dropout=0.5,recurrent_regularizer = R3,kernel_regularizer = R3,bias_regularizer = R3,
        name = 'lstm2')(lstm_output1)

        fc1 = Dense(fc1_layer, activation = 'relu', kernel_regularizer = R3,bias_regularizer = R3,name = 'FC1')
        z_out = fc1(lstm_output2)

        # Fc1 = wrappers.TimeDistributed(fc1,name = 'TFC1'+'-'+scale)
        # z_out = Fc1(Dropout(0.5,name='Dropout1'+'-'+scale)(lstm_output2))
        # z_out = fc1(Dropout(0.5)(lstm_output2))
        # z_out = fc1(BatchNormalization()(lstm_output2))
        # z_out = Activation('relu')(Dropout(0.5)(lstm_output2))
        

        
        ## temporal pooling layers
        # temporal_pooling_layer = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]),name ='Lambda2')
        # temporal_pooling = [0,1,2]
        # z_out_t = [0,0,0]
        # for pool_scale in range(len(temporal_pooling)):
        #     pool_size =video_length//(2**pool_scale)
        #     z_out_slice = []
        #     for seg in range(2**pool_scale-1):
        #         z_out_slice.append(temporal_pooling_layer(Lambda(lambda x: x[:,pool_size*seg:pool_size*(seg+1)])(z_out)))
        #     if pool_scale == 0:
        #         z_out_slice.append(temporal_pooling_layer(z_out))
        #         z_out_t[pool_scale] = z_out_slice[0] 
        #         continue
        #     else:
        #         z_out_slice.append(temporal_pooling_layer(Lambda(lambda x: x[:,pool_size*(seg+1):])(z_out)))
        #     z_out_t[pool_scale] = keras.layers.concatenate(z_out_slice,axis=-1) 
        # z_out_f = keras.layers.concatenate(z_out_t,axis=-1) 

        z_out_f = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]),name ='Lambda2')(lstm_output2)

        # fc2 = Dense(class_num,activation='linear',kernel_regularizer = R3,bias_regularizer = R3,name = 'FC2'+'-'+scale)
        # main_output = Activation(activation='softmax')(Dropout(0.5,name='Dropout2'+'-'+scale)(z_out_f))
        
        fc2 = Dense(class_num,activation='softmax',kernel_regularizer = R3,bias_regularizer = R3,name = 'FC2')
        # main_output = fc2(Dropout(0.5,name='Dropout2'+'-'+scale)(z_out_f))
        main_output = fc2(z_out_f)
    else: #prediction fusion
        ######### input to the three models
        fc_output1 = model_scale1(main_input1)
        fc_output2 = model_scale2(main_input2)
        fc_output3 = model_scale3(main_input3)
    
        main_output = keras.layers.average([fc_output1,fc_output2,fc_output3])
    
    model = Model(inputs=[main_input1,main_input2,main_input3], outputs=[main_output])
    model.summary()
    # model.load_weights('full_earlyfusion_model_ta.h5')
    # visualize_layer(model_scale3, 'max_pooling3d_9', x_train3[11:12], 24) # isualize_layer(model, layer_name, data, time_step)
    adam_sgd = op.Adam(amsgrad=True)
    sgd = op.SGD(lr = 0.00001,momentum = 0.9)
    model.compile(loss='categorical_crossentropy', optimizer = sgd,metrics=['accuracy'])
    ## using multiple GPU model
    mgpu_model = keras.utils.multi_gpu_model(model,gpus=2)
    mgpu_model.compile(loss='categorical_crossentropy', optimizer = sgd,metrics=['accuracy'])
    mgpu_model.summary()
    
    ###########################################################
    # first train the fusion model with adam_sgd
    start_time = time.time()
    print('jointly first fine-tune the whole fusion network with adam')
    model.compile(loss='categorical_crossentropy', optimizer = adam_sgd, metrics=['accuracy'])
    ## using multiple GPU model
    mgpu_model = keras.utils.multi_gpu_model(model,gpus=2)
    mgpu_model.compile(loss='categorical_crossentropy', optimizer = adam_sgd, metrics=['accuracy'])
    mgpu_model.summary()
    history1 = mgpu_model.fit([x_train1,x_train2,x_train3],y_train1, batch_size = batch_size, epochs=epochs[0]) #epochs=epochs[0] for adam
    model.save('full_earlyfusion_model_ta.h5')

    print('jointly second fine-tune the whole fusion network with sgd')
    start_time = time.time()
    model.compile(loss='categorical_crossentropy', optimizer = sgd,metrics=['accuracy'])
    ## using multiple GPU model
    mgpu_model = keras.utils.multi_gpu_model(model,gpus=2)
    mgpu_model.compile(loss='categorical_crossentropy', optimizer = sgd,metrics=['accuracy'])
    mgpu_model.summary()
    model.load_weights('full_earlyfusion_model_ta.h5') #60 adam_sgd
    lr_reducer = LrReducer()
    tensorboard = TensorBoard()
    model_filepath="early_model_{epoch:02d}-{val_acc:.2f}.h5"
    checkpoint = ModelCheckpoint('/home/data/logs/'+ model_filepath, monitor = 'val_acc', save_weights_only=True)
    history2 = mgpu_model.fit([x_train1,x_train2,x_train3],y_train1, batch_size = batch_size, epochs=epochs[1],validation_split = 0.2, callbacks=[lr_reducer,tensorboard,checkpoint]) #epochs=epochs[1] for sgd
 
    average_time_per_epoch = (time.time() - start_time) / epochs[1]
    print('Training duration (s) : ', time.time() - global_start_time)
    model.save('full_early_c3dmodel_final_cv.h5')
    # sub_model1 = model.layers[3]
    # sub_model2 = model.layers[4]
    # sub_model3 = model.layers[5]
    # weights1 = sub_model1.layers[4].get_weights()
    # weights2 = sub_model2.layers[4].get_weights()
    # weights3 = sub_model3.layers[4].get_weights()    
    # save_metric_matrix(weights1[0], scale ='1')
    # save_metric_matrix(weights2[0], scale ='2')
    # save_metric_matrix(weights3[0], scale ='3')
    
    
    scores = mgpu_model.evaluate([x_test1,x_test2,x_test3],y_test1, batch_size = batch_size)
    print('Test score:', scores[0])
    print('Test accuracy:', scores[1])
    print('sub_model performance:',test_score)
    print('Training duration (s) : ', time.time() - global_start_time)
    
    plot_model(model,to_file='model.png')
    
    
    # y_label = []
    # global_memory_list = []
    # [y_label.append(y_train.argmax(y_label[i])) for i in range(y_train.shape[0])]
    # for i in range(np.unique(y_label).shape): 
        # global_memory_list.append(np.mean(memory_output[np.array(y_label) == i,:],axis=1))
     
    y_pred = mgpu_model.predict([x_test1,x_test2,x_test3])
    matrix = confusion_matrix(y_test1.argmax(axis=1), y_pred.argmax(axis=1))
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    # np.set_printoptions(precision=2)
    # plt.figure()
    # plot_confusion_matrix(matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
    # plt.savefig("confusion_matrix.png")

    file1 = h5py.File('matrix_ntu.h5','w')
    file1.create_dataset('matrix', data = matrix)
    file1.close()
    
    #plot_results(predicted,Y_test,'predicted results')

    # Compare models' accuracy, loss and elapsed time per epoch.
    # history = joblib.load('ntu_history.pkl')
    # history = history1
    plt.style.use('ggplot')
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax1.set_title('Model accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.legend(['Train', 'Test'], loc='upper left')
    ax1.plot(history1.history['acc']+history2.history['acc'])
    
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epochs')
    ax2.legend(['Train', 'Test'], loc='upper left')
    history.history['loss'].append(history2.history['loss'])
    ax2.plot(history1.history['loss']+history2.history['loss'])
    
    ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    ax3.set_title('Loss vs Accuracy')
    ax3.set_xlabel('Loss')
    ax3.set_ylabel('Accuracy')
    ax3.plot(history1.history['loss']+history2.history['loss'],history1.history['acc']+history2.history['acc'])
    
    plt.tight_layout()
    plt.savefig("ntu_training.png")

    
    # predicted = model.predict([X_test,X_test,X_test_1])
    # print('predicted shape:',np.array(predicted).shape)  #(16488L,1L)
    # predicted = np.reshape(predicted, (predicted.size,)) #(16488L,)
	
  
