'''implement the MSNN on NTU RGB+D dataset '''
# python conv-lstm-ntu-fusion-v2-ta-pretrain_spp_metric.py
import time
import numpy as np
import keras
import h5py
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, LSTM, Permute,RepeatVector, Dropout, Input, wrappers, Activation, MaxPooling3D, RNN, ActivityRegularization, Masking,Lambda,Conv3D,MaxPooling2D,GlobalMaxPooling2D,GRU,BatchNormalization
from MetricLayer import MetricLayer
from keras.utils import plot_model
from keras import regularizers
from keras import backend as K
from keras.models import load_model
import keras.optimizers as op
import random
from keras_spp.spp.SpatialPyramidPooling import SpatialPyramidPooling3D,SpatialPyramidPooling
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint,TensorBoard
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 
## padding the sequence to the same length by uniformly sampling the sequence
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
    '''Padding the human pose in those action samples which only have a single human pose appeared in video,
       because some actions are performed by one single subject, while other actions are performed by two subjects.
    '''
    assert len(x.shape) == 2
    joints_xyz = x.shape[1]
    for n in range(x.shape[0]):
        if all(x[n,joints_xyz//2:] == 0):
            x[n,joints_xyz//2:] = x[n,:joints_xyz//2]
    return x        

def load_data(filepath, max_len,scale):

    ''' load and transform the raw skeleton data from *.mat files created by Matlab.
    '''
    
    ## First, load the training data#########################################
    f1 = h5py.File(filepath+'data_cv3_scale'+scale+'.mat', 'r')
    train_data = [f1[element] for element in f1['data_cv3'][0]] 
    train_label = [f1[element] for element in f1['data_cv3'][1]]

    
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
    
    ## Second, load the testing data#########################################
    f2 = h5py.File(filepath+'data_cv1_scale'+scale+'.mat', 'r')
    test_data = [f2[element] for element in f2['data_cv1'][0]] 
    test_label = [f2[element] for element in f2['data_cv1'][1]] 
    
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

    ## padding the sequences with various lengths#####################
    print('Pad sequences (samples x time)')
    x_train = pad_sequences(train_data,max_len)
    x_test = pad_sequences(test_data,max_len)
    # x_train = sequence.pad_sequences(train_data, maxlen = max_len, dtype = 'float32', padding = 'post',truncating='post')
    # x_test = sequence.pad_sequences(test_data, maxlen = max_len,  dtype = 'float32', padding = 'post',truncating='post')
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)   
    return [x_train, y_train, x_test, y_test]
    
def morph_data(filepath,max_len,scale):

    '''reshape the raw data to the shape of
       dim 0: the number of samples
       dim 1: the number of frames
       dim 2: the number of joints
       dim 3: 3 (x,y,z) 

       outputs are saved as *.h5 files
    '''

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
    
# def distribute_alpha_t(x):
#     return K.repeat_elements(x,3,axis=2)
# def distribute_alpha_t_output_shape(input_shape):
#     shape = list(input_shape)
#     assert len(shape) == 3  # only valid for 3D tensors
#     shape= [input_shape[0],input_shape[1],input_shape[2]*3]
#     return tuple(shape)

def flattenConv(x):
    # shape = x.shape.as_list()
    # dim = np.prod(shape[2:])
    shape = K.shape(x)
    dim = K.prod(shape[2:])
    return K.reshape(x,[shape[0],shape[1],dim])
def flattenConv_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 5  # only valid for 5D tensors
    shape= [input_shape[0],input_shape[1],input_shape[2]*input_shape[3]*input_shape[4]]
    return tuple(shape)  
    
# def unflattenConv(x):
#     shape = K.shape(x)
#     # shape = x.shape.as_list()
#     return K.reshape(x,[-1,shape[1],K.cast(K.sqrt(shape[2]),dtype='int32'),K.cast(K.sqrt(shape[2]),dtype='int32'),1])
# def unflattenConv_output_shape(input_shape):
#     shape = list(input_shape)
#     assert len(shape) == 3  # only valid for 3D tensors
#     shape= [input_shape[0],input_shape[1],K.cast(K.sqrt(input_shape[2]),dtype='int32'),K.cast(K.sqrt(input_shape[2]),dtype='int32'),1]
#     return tuple(shape)     

def flattenSSM(x):
    shape = x.shape.as_list()
    # dim = np.prod(shape[2:])
    # shape = K.shape(x)
    return K.reshape(x,[-1,shape[1], int(np.sqrt(shape[2])),int(np.sqrt(shape[2])), 1])
def flattenSSM_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4  # only valid for 4D tensors
    shape= [input_shape[0],input_shape[1],int(np.sqrt(input_shape[2])),int(np.sqrt(input_shape[2])),1]
    return tuple(shape)

##  pairwise joint graph building by following 4 custom functions.    
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


def save_metric_matrix(weights, scale ='3'):
    file1 = h5py.File('Metric_Matrix'+scale+'.h5','w')
    file1.create_dataset('matrix', data = weights)
    file1.close()

class LrReducer(keras.callbacks.Callback):
    '''
    training strategy mentioned in the paper
    '''
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


def get_model(layers_num,joint_num, max_len, stride_set,class_num, reg,spp_numbers,spp_numbers_a,scale):
    
    ## parameter setting
    clip_length = 35
    main_layers = layers_num[0:5] 
    fc1_layer = layers_num[5]
    conv_feats_dim = layers_num[6]
    lamda4 = 0.00001
    lamda4 = 0.0
    
    ## define the input shape
    main_input = Input(shape=(max_len,main_layers[0],3),dtype = 'float32', name = 'main_input'+'-'+scale)
    repeat_input1 = Lambda(repeat_x_onejoint,repeat_x_onejoint_output_shape)(main_input)
    repeat_input2 = Lambda(repeat_x_groupjoint,repeat_x_groupjoint_output_shape)(main_input)
    ########define the model###########
    ssm_input = keras.layers.subtract([repeat_input1,repeat_input2])
    ssm_input = MetricLayer(joint_num,kernel_regularizer = regularizers.Maha_R(lamda4))(ssm_input)
    ssm_input = Lambda(flattenSSM,flattenSSM_output_shape)(ssm_input)
    c3d_out1 = Conv3D(main_layers[1],kernel_size=[3,3,3],strides = stride_set,activation = 'relu', kernel_regularizer=reg, bias_regularizer=reg,activity_regularizer=reg)(ssm_input)
    # c3d_out1 = MaxPooling3D(pool_size=(1, 2, 2),strides = (1,2,2))(c3d_out1)
    c3d_out2 = Conv3D(main_layers[2],kernel_size=[3,3,3],strides = stride_set,activation = 'relu', kernel_regularizer=reg, bias_regularizer=reg,activity_regularizer=reg)(c3d_out1)
    if scale == '1':
        c3d_out2 = MaxPooling3D(pool_size=(2, 1, 1),strides = (2,1,1))(c3d_out2)
        c3d_out3 = Conv3D(main_layers[3],kernel_size=[3,3,3],strides = stride_set,activation = 'relu',kernel_regularizer=reg, bias_regularizer=reg,activity_regularizer=reg)(c3d_out2)
        c3d_out3 = MaxPooling3D(pool_size=(2, 1, 1),strides = (2,1,1))(c3d_out3)
    if scale == '2':
        c3d_out2 = MaxPooling3D(pool_size=(2, 2, 2),strides = (2,2,2))(c3d_out2)
        c3d_out3 = Conv3D(main_layers[3],kernel_size=[3,3,3],strides = stride_set,activation = 'relu',kernel_regularizer=reg, bias_regularizer=reg,activity_regularizer=reg)(c3d_out2)
        c3d_out3 = MaxPooling3D(pool_size=(2, 1, 1),strides = (2,1,1))(c3d_out3)
    if scale == '3':
        c3d_out2 = MaxPooling3D(pool_size=(2, 2, 2),strides = (2,2,2))(c3d_out2)
        c3d_out3 = Conv3D(main_layers[3],kernel_size=[3,3,3],strides = stride_set,activation = 'relu',kernel_regularizer=reg, bias_regularizer=reg,activity_regularizer=reg)(c3d_out2)
        c3d_out3 = MaxPooling3D(pool_size=(2, 2, 2),strides = (2,2,2))(c3d_out3)
    
    spp_layer1 = wrappers.TimeDistributed(SpatialPyramidPooling(spp_numbers_a),name = 'TSPP1'+'-'+scale) 
    t_atten = spp_layer1(ssm_input)
    
    # t_atten = Lambda(flattenConv,flattenConv_output_shape)(main_input)
    fc1_ta = Dense(main_layers[4]//2, activation='relu',kernel_regularizer = reg,bias_regularizer = reg,name = 'FC1_ta')
    t_atten = fc1_ta(t_atten)
    t_atten = Permute((2, 1))(t_atten)
    fc2_ta = Dense(clip_length, activation='relu',kernel_regularizer = reg,bias_regularizer = reg,name = 'FC2_ta')
    t_atten = fc2_ta(t_atten)
    t_atten = Permute((2, 1))(t_atten)
    fc3_ta = Dense(conv_feats_dim, activation='sigmoid',kernel_regularizer = reg,bias_regularizer = reg,name = 'FC3_ta')
    t_atten = fc3_ta(t_atten)

    
    
    spp_layer2 = wrappers.TimeDistributed(SpatialPyramidPooling(spp_numbers),name = 'TSPP2'+'-'+scale) 
    lstm_input = spp_layer2(c3d_out3)
    # lstm_input = Lambda(flattenConv,flattenConv_output_shape)(c3d_out3)

    lstm_input = keras.layers.multiply([t_atten,lstm_input])
    lstm1 = LSTM(main_layers[4],return_sequences=True, dropout=0.5,recurrent_regularizer = reg,kernel_regularizer = reg,bias_regularizer = reg,
    name = 'lstm1'+'-'+scale)
    lstm_output1 = lstm1(lstm_input)
    lstm2 = LSTM(main_layers[4],return_sequences=True, dropout=0.5,recurrent_regularizer = reg,kernel_regularizer = reg,bias_regularizer = reg,
    name = 'lstm2'+'-'+scale)
    lstm_output2 = lstm2(lstm_output1)
 
    fc1 = Dense(fc1_layer,activation='relu', kernel_regularizer = reg,bias_regularizer = reg,name = 'FC1'+'-'+scale)
    z_out = fc1(lstm_output2)
    z_out_f = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]),name ='Lambda2'+'-'+scale)(lstm_output2)
    
    fc2 = Dense(class_num,activation='softmax',kernel_regularizer = reg,bias_regularizer = reg,name = 'FC2'+'-'+scale)
    main_output = fc2(z_out_f)
    model = Model(inputs=[main_input], outputs=[main_output])
    return model

def train_lstms(layers_num,batch_size,joint_num,reg,filepath,stride_set,spp_numbers,spp_numbers_a,epoch,fine_tune = False,scale='1'):
    global_start_time = time.time()
    max_len = 150
    class_num = 60
    results=[]
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
    
    ################define the model############################
    model = get_model(layers_num,joint_num, max_len, stride_set,class_num,reg,spp_numbers,spp_numbers_a,scale)
    if fine_tune == True:
        model.load_weights('model_scale3.h5')
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
    print('Training duration (s) : ', time.time() - global_start_time)
    ################evaluate the model by using the test set##############################
    scores = mgpu_model.evaluate(x_test,y_test, batch_size = batch_size)
    print('Test score:', scores[0])
    print('Test accuracy:', scores[1])
    print('Training duration (s) : ', time.time() - global_start_time)
    # plot_model(model,to_file='model.png') 

    return [model,scores[1]]

if __name__=='__main__':
    global_start_time = time.time()
    epochs = [60,80]
    max_len = 150
    class_num = 60
    lamda1 = 0.01
    lamda2 = 0.001
    lamda3 = 0.000001
    test_score=[0,0,0]
    results = []
    stride_set = [1,1,1]
    reg = regularizers.l2(lamda3)
    batch_size = 16
    filepath = '/home/data/nturgbd_skeletons/ntu_data_mat/'

    print('morphing the data into [samples, t, joint_num,3] format....')
    morph_data(filepath, max_len,scale ='1') 
    morph_data(filepath, max_len,scale ='2') 
    morph_data(filepath, max_len,scale ='3')
    
    ###########################################################
    joint_num1  = 7*2
    spp_numbers1 = [4] # sum([ x**2 for x in spp_numbers1 ])*16 for the spp layer
    spp_numbers1_a = [7]
    layers_num1 = [joint_num1,4,8,16,100,100,sum([ x**2 for x in spp_numbers1 ])*16] ## 16,32,64,100##64,64,64,200 [input_layer,lstm1-lstm3,fc1_layer,fc2_layer] [joint_num1,50,50,50,100]
    model_scale1,test_score[0]=train_lstms(layers_num1,batch_size,joint_num1,reg,filepath,stride_set,spp_numbers1,spp_numbers1_a,epoch = 40,fine_tune = False,scale='1')
    model_scale1.save('model_scale1.h5')    
    #########################################################
    joint_num2  = 12*2
    spp_numbers2 = [4]
    spp_numbers2_a = [12]
    layers_num2 = [joint_num2,8,16,32,100,100,sum([ x**2 for x in spp_numbers2 ])*32] ##128, 32,32,100,100##64,64,64,200 [input_layer,lstm1-lstm3,fc1_layer,fc2_layer] [joint_num2,80,80,80,160]
    model_scale2,test_score[1]=train_lstms(layers_num2,batch_size,joint_num2,reg,filepath,stride_set,spp_numbers2,spp_numbers2_a,epoch=40,fine_tune = False,scale='2')
    model_scale2.save('model_scale2.h5')
    ##########################################################
    joint_num3  = 25*2
    spp_numbers3 = [4]
    spp_numbers3_a = [20]
    layers_num3 = [joint_num3,16,32,64,100,100,sum([ x**2 for x in spp_numbers3 ])*64] ##576, 32,64,64,200 [input_layer,lstm1-lstm3,fc1_layer,fc2_layer] [joint_num2,100,100,100,200]
    model_scale3,test_score[2]=train_lstms(layers_num3,batch_size,joint_num3,reg,filepath,stride_set,spp_numbers3,spp_numbers3_a, epoch= 40,fine_tune = False,scale='3')
    model_scale3.save('model_scale3.h5')
    ###########################################################
    model_scale1 = get_model(layers_num1,joint_num1, max_len, stride_set,class_num, reg,spp_numbers1,spp_numbers1_a,scale = '1')
    model_scale2 = get_model(layers_num2,joint_num2, max_len, stride_set,class_num, reg,spp_numbers2,spp_numbers2_a,scale = '2')
    model_scale3 = get_model(layers_num3,joint_num3, max_len, stride_set,class_num, reg,spp_numbers3,spp_numbers3_a,scale = '3')
    model_scale1.load_weights('model_scale1.h5')
    model_scale2.load_weights('model_scale2.h5')
    model_scale3.load_weights('model_scale3.h5')
    model_scale1.name ='model_1'
    model_scale2.name ='model_2'
    model_scale3.name ='model_3' # to avoid name conflicts of different models
    #####define the inputs
    main_input1= Input(shape=(max_len,joint_num1,3), dtype = 'float32')
    main_input2= Input(shape=(max_len,joint_num2,3), dtype = 'float32')	
    main_input3= Input(shape=(max_len,joint_num3,3), dtype = 'float32')
    
    ####load the data of varying scales#########################
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

    if 1: # feature fusion
        
        ### redirect the outputs####
        sub_model1 = Model(inputs = model_scale1.inputs, outputs = model_scale1.layers[-2].output)
        sub_model2 = Model(inputs = model_scale2.inputs, outputs = model_scale2.layers[-2].output)
        sub_model3 = Model(inputs = model_scale3.inputs, outputs = model_scale3.layers[-2].output)
        
        ######### input to the three models
        fc_output1 = sub_model1(main_input1)
        fc_output2 = sub_model2(main_input2)
        fc_output3 = sub_model3(main_input3)
        
              
        fc_combined = keras.layers.concatenate([fc_output1,fc_output2,fc_output3]) 
        # fc_combined = keras.layers.maximum([fc_output1,fc_output2,fc_output3])
        # fc_combined = keras.layers.add([fc_output1,fc_output2,fc_output3])      
        fc_combined_bn = BatchNormalization(name='BN_layer')(fc_combined)
        fc_softmax  = Dense(class_num,activation='softmax',kernel_regularizer = reg,bias_regularizer = reg,name = 'FC3')
        main_output = fc_softmax(Dropout(0.5)(fc_combined_bn))
    else: #prediction fusion
        ######### input to the three models
        fc_output1 = model_scale1(main_input1)
        fc_output2 = model_scale2(main_input2)
        fc_output3 = model_scale3(main_input3)
    
        main_output = keras.layers.average([fc_output1,fc_output2,fc_output3])
    
    model = Model(inputs=[main_input1,main_input2,main_input3], outputs=[main_output])
    model.summary()
    
    ###########################################################
    # first train the fusion model with adam_sgd
    start_time = time.time()
    print('jointly first fine-tune the whole fusion network with adam')
    adam_sgd = op.Adam(amsgrad=True)
    model.compile(loss='categorical_crossentropy', optimizer = adam_sgd,metrics=['accuracy'])
    ## using multiple GPU model
    mgpu_model = keras.utils.multi_gpu_model(model,gpus=2)
    mgpu_model.compile(loss='categorical_crossentropy', optimizer = adam_sgd,metrics=['accuracy'])
    mgpu_model.summary()
    history = mgpu_model.fit([x_train1,x_train2,x_train3],y_train1, batch_size = batch_size, epochs=epochs[0])
    model.save('full_fusion_model.h5')

    # sencond fine-tune the fusion model with sgd
    print('jointly second fine-tune the whole fusion network with sgd')
    start_time = time.time()
    sgd = op.SGD(lr = 0.0001,momentum = 0.9)
    model.compile(loss='categorical_crossentropy', optimizer = sgd,metrics=['accuracy'])
    ## using multiple GPU model
    mgpu_model = keras.utils.multi_gpu_model(model,gpus=2)
    mgpu_model.compile(loss='categorical_crossentropy', optimizer = sgd,metrics=['accuracy'])
    mgpu_model.summary()
    model.load_weights('full_fusion_model.h5') #40 adam
    lr_reducer = LrReducer()
    tensorboard = TensorBoard()
    model_filepath="late_model_{epoch:02d}-{val_acc:.2f}.h5"
    checkpoint = ModelCheckpoint('./logs/'+ model_filepath, monitor = 'val_acc', save_weights_only=True)
    history = mgpu_model.fit([x_train1,x_train2,x_train3],y_train1, batch_size = batch_size, epochs=epochs[1],validation_split = 0.2, callbacks=[lr_reducer,tensorboard,checkpoint])

    average_time_per_epoch = (time.time() - start_time) / epochs[0]
    print('Training duration (s) : ', time.time() - global_start_time)
    model.save('full_latefusion_model_final.h5')
    model.layers[-5].save('late_model_scale3_ptrained.h5')
    model.layers[-6].save('late_model_scale2_ptrained.h5')
    model.layers[-7].save('late_model_scale1_ptrained.h5')
    
    
    scores = mgpu_model.evaluate([x_test1,x_test2,x_test3],y_test1, batch_size = batch_size)
    print('Test score:', scores[0])
    print('Test accuracy:', scores[1])
    print('sub_model performance:',test_score)
    print('Training duration (s) : ', time.time() - global_start_time)
    
    plot_model(model,to_file='model.png')
    
   
     

    # Compare models' accuracy, loss and elapsed time per epoch.
    plt.style.use('ggplot')
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax1.set_title('Model accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.legend(['Train', 'Test'], loc='upper left')
    ax1.plot(history.history['acc'])
    
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epochs')
    ax2.legend(['Train', 'Test'], loc='upper left')
    ax2.plot(history.history['loss'])
    
    ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    ax3.set_title('Loss vs Accuracy')
    ax3.set_xlabel('Loss')
    ax3.set_ylabel('Accuracy')
    ax3.plot(history.history['loss'],history.history['acc'])
    
    plt.tight_layout()
    # plt.show()
    plt.savefig("pre_training.png")

    
    # predicted = model.predict([X_test,X_test,X_test_1])
    # print('predicted shape:',np.array(predicted).shape)  #(16488L,1L)
    # predicted = np.reshape(predicted, (predicted.size,)) #(16488L,)
	
  
