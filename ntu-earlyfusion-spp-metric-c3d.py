'''MSNN_early-C3D -- early-fusion MSNN using pretrained C3D networks as the
3D CNN branches, on the NTU RGB+D dataset.

    python ntu-earlyfusion-spp-metric-c3d.py

Instead of the light-weight CNNs of `ntu-earlyfusion-spp-metric.py`, each
stream here is the Sports-1M pretrained C3D network (Sec. III-B).  Download
`sports1M_weights_tf.h5` first -- see the README.
'''
import time
import keras
import keras.optimizers as op
from keras.models import Model, Sequential
from keras.layers import (BatchNormalization, Dense, Dropout, Flatten, Input,
                          Lambda, LSTM)
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard

from MetricLayer import MetricLayer_ForC3D
from data_utils import load_raw, morph_data, save_metric_matrix
from ssi_layers import (flattenConv, flattenConv_output_shape,
                        flattenSSM3, flattenSSM3_output_shape,
                        repeat_x_groupjoint, repeat_x_groupjoint_output_shape,
                        repeat_x_onejoint, repeat_x_onejoint_output_shape)
from viz_utils import LrReducer, plot_training_history
## For inspecting SSIs / filter responses, see viz_utils.visualize_layer.

## ---------------------------------------------------------------------------
## Configuration -- edit FILEPATH to point at your prepared *.mat / *.h5 data.
## ---------------------------------------------------------------------------
FILEPATH = '/home/data/nturgbd_skeletons/ntu_data_mat/'

GPUS = 2               # number of GPUs for keras.utils.multi_gpu_model
MAX_LEN = 150          # frames per sequence after padding/subsampling
CLASS_NUM = 60         # NTU RGB+D has 60 action classes
CLIP_LENGTH = 35       # temporal length after the 3D CNN downsampling
BATCH_SIZE = 16        # Sec. IV-B
LAMBDA1 = 0.000001     # l2 coefficient of Eq. (8); Sec. IV-B sets it to 1e-6



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

if __name__=='__main__':
    global_start_time = time.time()
    epochs = [60,60]
    max_len = MAX_LEN
    class_num = CLASS_NUM
    stride_set = [1,1,1]
    reg = regularizers.l2(LAMBDA1)
    batch_size = BATCH_SIZE
    filepath = FILEPATH

    print('morphing the data into [samples, t, joint_num,3] format....')
    morph_data(filepath, max_len, scale='1')
    morph_data(filepath, max_len, scale='2')
    morph_data(filepath, max_len, scale='3')

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
    ## MetricLayer_ForC3D replicates the SSI into 3 channels, which is what the
    ## Sports-1M pretrained C3D backbone expects.  (The plain single-channel
    ## MetricLayer used to be wired in here, which does not match the 3-channel
    ## `conv1` weights loaded above.)
    ## Eq. (8) folds the metric weights L into the global l2 term ||W||_2, so
    ## the same regulariser is used here.  Pass `kernel_regularizer=None`
    ## instead to reproduce the unregularised L of the original runs.
    ## Each stream gets its own joint count (the previous code referenced an
    ## undefined `joint_num` here).
    ssm_input1 = MetricLayer_ForC3D(joint_num1,kernel_regularizer = reg,
                                    name = 'metric_layer-1')(ssm_input1)
    ssm_input1 = Lambda(flattenSSM3,flattenSSM3_output_shape)(ssm_input1)
    ssm_input2 = MetricLayer_ForC3D(joint_num2,kernel_regularizer = reg,
                                    name = 'metric_layer-2')(ssm_input2)
    ssm_input2 = Lambda(flattenSSM3,flattenSSM3_output_shape)(ssm_input2)
    ssm_input3 = MetricLayer_ForC3D(joint_num3,kernel_regularizer = reg,
                                    name = 'metric_layer-3')(ssm_input3)
    ssm_input3 = Lambda(flattenSSM3,flattenSSM3_output_shape)(ssm_input3)

    ####load the data of varying scales#########################
    x_train1, y_train1, x_test1, y_test1 = load_raw(filepath, '1')
    x_train2, y_train2, x_test2, y_test2 = load_raw(filepath, '2')
    x_train3, y_train3, x_test3, y_test3 = load_raw(filepath, '3')

    print('x_train shape:', x_train1.shape,x_train2.shape,x_train3.shape)  #(40091L, 150L,rows,cols,channels)
    print('y_train shape:', y_train1.shape,y_train2.shape,y_train3.shape)  #(40091L,60L)
    print('x_test shape:',  x_test1.shape,x_test2.shape,x_test3.shape)   #(16487L, 150L,rows,cols,channels)
    print('y_test shape',   y_test1.shape,y_test2.shape,y_test3.shape)   #(16487L,60L)

    ######### input to the three models
    c3d_out1 = model_scale1(ssm_input1)
    c3d_out2 = model_scale2(ssm_input2)
    c3d_out3 = model_scale3(ssm_input3)
    
    feat_output1 = Lambda(flattenConv,flattenConv_output_shape)(c3d_out1)
    feat_output2 = Lambda(flattenConv,flattenConv_output_shape)(c3d_out2) 
    feat_output3 = Lambda(flattenConv,flattenConv_output_shape)(c3d_out3)      
    feat_combined = keras.layers.concatenate([feat_output1,feat_output2,feat_output3])       
    feat_combined_bn = BatchNormalization(name='BN_layer')(feat_combined)
    lstm_output1 = LSTM(lstm_layer_num,return_sequences=True, dropout=0.5,recurrent_regularizer = reg,kernel_regularizer = reg,bias_regularizer = reg,
    name = 'lstm1')(feat_combined_bn)
    lstm_output2 = LSTM(lstm_layer_num,return_sequences=True, dropout=0.5,recurrent_regularizer = reg,kernel_regularizer = reg,bias_regularizer = reg,
    name = 'lstm2')(lstm_output1)

    z_out_f = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]),name ='Lambda2')(lstm_output2)
    fc_softmax = Dense(class_num,activation='softmax',kernel_regularizer = reg,bias_regularizer = reg,name = 'FC2')
    main_output = fc_softmax(z_out_f)
    
    model = Model(inputs=[main_input1,main_input2,main_input3], outputs=[main_output])
    model.summary()
    ## To inspect a stream's feature maps, e.g.:
    ##   from viz_utils import visualize_layer
    ##   visualize_layer(model_scale3, 'max_pooling3d_9', x_train3[11:12], 24)
    adam_sgd = op.Adam(amsgrad=True)
    sgd = op.SGD(lr = 0.00001,momentum = 0.9)

    ###########################################################
    # Stage 1: train the fusion model with Adam (Sec. IV-B)
    print('jointly first fine-tune the whole fusion network with adam')
    model.compile(loss='categorical_crossentropy', optimizer = adam_sgd, metrics=['accuracy'])
    ## using multiple GPU model
    mgpu_model = keras.utils.multi_gpu_model(model,gpus=GPUS)
    mgpu_model.compile(loss='categorical_crossentropy', optimizer = adam_sgd, metrics=['accuracy'])
    mgpu_model.summary()
    history1 = mgpu_model.fit([x_train1,x_train2,x_train3],y_train1, batch_size = batch_size, epochs=epochs[0]) #epochs=epochs[0] for adam
    ## Checkpoint marking the Adam -> SGD stage boundary.
    model.save('full_earlyfusion_c3dmodel.h5')

    # Stage 2: continue fine-tuning with SGD (Sec. IV-B)
    print('jointly second fine-tune the whole fusion network with sgd')
    start_time = time.time()
    model.compile(loss='categorical_crossentropy', optimizer = sgd,metrics=['accuracy'])
    ## using multiple GPU model
    mgpu_model = keras.utils.multi_gpu_model(model,gpus=GPUS)
    mgpu_model.compile(loss='categorical_crossentropy', optimizer = sgd,metrics=['accuracy'])
    mgpu_model.summary()
    lr_reducer = LrReducer()
    tensorboard = TensorBoard()
    model_filepath="early_model_{epoch:02d}-{val_acc:.2f}.h5"
    checkpoint = ModelCheckpoint('/home/data/logs/'+ model_filepath, monitor = 'val_acc', save_weights_only=True)
    history2 = mgpu_model.fit([x_train1,x_train2,x_train3],y_train1, batch_size = batch_size, epochs=epochs[1],validation_split = 0.2, callbacks=[lr_reducer,tensorboard,checkpoint]) #epochs=epochs[1] for sgd

    print('Training duration (s) : ', time.time() - start_time)
    model.save('full_earlyfusion_c3dmodel_final.h5')
    ## Dump each stream's learned Mahalanobis transform L.
    for scale in ('1', '2', '3'):
        weights = model.get_layer('metric_layer-' + scale).get_weights()
        save_metric_matrix(weights[0], scale=scale)

    scores = mgpu_model.evaluate([x_test1,x_test2,x_test3],y_test1, batch_size = batch_size)
    print('Test score:', scores[0])
    print('Test accuracy:', scores[1])
    print('Training duration (s) : ', time.time() - global_start_time)

    # Compare models' accuracy and loss per epoch across both training stages.
    plot_training_history([history1, history2], 'ntu_c3d_training.png')
