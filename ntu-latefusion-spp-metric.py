'''MSNN_late -- late-fusion Multi-Stream Neural Network on the NTU RGB+D dataset.

    python ntu-latefusion-spp-metric.py

Each of the three streams is trained on SSIs of one scale, then the streams are
fused at the feature level and fine-tuned jointly (Adam, then SGD) as described
in Sec. III-B3 and Sec. IV-B of the paper.
'''
import time
import keras
import keras.optimizers as op
from keras.models import Model
from keras.layers import (BatchNormalization, Conv3D, Dense, Dropout, Input,
                          Lambda, LSTM, MaxPooling3D, Permute, wrappers)
from keras.utils import plot_model
from keras import regularizers
from keras import backend as K
from keras_spp.spp.SpatialPyramidPooling import SpatialPyramidPooling
from keras.callbacks import ModelCheckpoint, TensorBoard

from MetricLayer import MetricLayer
from data_utils import load_raw, morph_data
from ssi_layers import (flattenSSM, flattenSSM_output_shape,
                        repeat_x_groupjoint, repeat_x_groupjoint_output_shape,
                        repeat_x_onejoint, repeat_x_onejoint_output_shape)
from viz_utils import LrReducer, plot_training_history

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

## The late-fusion protocol reads its cross-view splits under the protocol
## names produced by the Matlab preprocessing.
MAT_NAMES = dict(train_mat='data_cv3_scale', train_key='data_cv3',
                 test_mat='data_cv1_scale', test_key='data_cv1')

## 'feature' concatenates the per-stream FC features before the softmax;
## 'prediction' averages the three per-stream softmax outputs instead.
FUSION_MODE = 'feature'

## Per-scale MaxPooling3D configuration (cf. TABLE I).  Each entry is the
## (pool_after_conv2, pool_after_conv3) pair of pool sizes; strides match.
POOL_CONFIG = {
    '1': ((2, 1, 1), (2, 1, 1)),
    '2': ((2, 2, 2), (2, 1, 1)),
    '3': ((2, 2, 2), (2, 2, 2)),
}

def get_model(layers_num,joint_num, max_len, stride_set,class_num, reg,spp_numbers,spp_numbers_a,scale):
    
    ## parameter setting
    main_layers = layers_num[0:5]
    conv_feats_dim = layers_num[6]

    ## define the input shape
    main_input = Input(shape=(max_len,main_layers[0],3),dtype = 'float32', name = 'main_input'+'-'+scale)
    repeat_input1 = Lambda(repeat_x_onejoint,repeat_x_onejoint_output_shape)(main_input)
    repeat_input2 = Lambda(repeat_x_groupjoint,repeat_x_groupjoint_output_shape)(main_input)
    ########define the model###########
    ssm_input = keras.layers.subtract([repeat_input1,repeat_input2])
    ## Eq. (8) folds the metric weights L into the global l2 term ||W||_2, so
    ## the same regulariser is used here.  Pass `kernel_regularizer=None`
    ## instead to reproduce the unregularised L of the original runs.
    ssm_input = MetricLayer(joint_num, kernel_regularizer=reg,
                            name='metric_layer' + '-' + scale)(ssm_input)
    ssm_input = Lambda(flattenSSM,flattenSSM_output_shape)(ssm_input)

    pool2, pool3 = POOL_CONFIG[scale]
    c3d_out1 = Conv3D(main_layers[1],kernel_size=[3,3,3],strides = stride_set,activation = 'relu', kernel_regularizer=reg, bias_regularizer=reg,activity_regularizer=reg)(ssm_input)
    c3d_out2 = Conv3D(main_layers[2],kernel_size=[3,3,3],strides = stride_set,activation = 'relu', kernel_regularizer=reg, bias_regularizer=reg,activity_regularizer=reg)(c3d_out1)
    c3d_out2 = MaxPooling3D(pool_size=pool2, strides=pool2)(c3d_out2)
    c3d_out3 = Conv3D(main_layers[3],kernel_size=[3,3,3],strides = stride_set,activation = 'relu',kernel_regularizer=reg, bias_regularizer=reg,activity_regularizer=reg)(c3d_out2)
    c3d_out3 = MaxPooling3D(pool_size=pool3, strides=pool3)(c3d_out3)

    spp_layer1 = wrappers.TimeDistributed(SpatialPyramidPooling(spp_numbers_a),name = 'TSPP1'+'-'+scale) 
    t_atten = spp_layer1(ssm_input)

    fc1_ta = Dense(main_layers[4]//2, activation='relu',kernel_regularizer = reg,bias_regularizer = reg,name = 'FC1_ta')
    t_atten = fc1_ta(t_atten)
    t_atten = Permute((2, 1))(t_atten)
    fc2_ta = Dense(CLIP_LENGTH, activation='relu',kernel_regularizer = reg,bias_regularizer = reg,name = 'FC2_ta')
    t_atten = fc2_ta(t_atten)
    t_atten = Permute((2, 1))(t_atten)
    fc3_ta = Dense(conv_feats_dim, activation='sigmoid',kernel_regularizer = reg,bias_regularizer = reg,name = 'FC3_ta')
    t_atten = fc3_ta(t_atten)

    spp_layer2 = wrappers.TimeDistributed(SpatialPyramidPooling(spp_numbers),name = 'TSPP2'+'-'+scale)
    lstm_input = spp_layer2(c3d_out3)

    lstm_input = keras.layers.multiply([t_atten,lstm_input])
    lstm1 = LSTM(main_layers[4],return_sequences=True, dropout=0.5,recurrent_regularizer = reg,kernel_regularizer = reg,bias_regularizer = reg,
    name = 'lstm1'+'-'+scale)
    lstm_output1 = lstm1(lstm_input)
    lstm2 = LSTM(main_layers[4],return_sequences=True, dropout=0.5,recurrent_regularizer = reg,kernel_regularizer = reg,bias_regularizer = reg,
    name = 'lstm2'+'-'+scale)
    lstm_output2 = lstm2(lstm_output1)
 
    ## The released architecture sums the LSTM outputs over time to form the
    ## stream representation u; the FC1 layer that used to be built here was
    ## never wired into the graph, so it is omitted.  `fc1_layer`
    ## (= layers_num[5]) is therefore unused in this model.
    z_out_f = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]),name ='Lambda2'+'-'+scale)(lstm_output2)


    fc2 = Dense(class_num,activation='softmax',kernel_regularizer = reg,bias_regularizer = reg,name = 'FC2'+'-'+scale)
    main_output = fc2(z_out_f)
    model = Model(inputs=[main_input], outputs=[main_output])
    return model

def train_stream(layers_num,batch_size,joint_num,reg,filepath,stride_set,spp_numbers,spp_numbers_a,epoch,fine_tune = False,scale='1'):
    global_start_time = time.time()

    print('> Loading data... ')
    x_train, y_train, x_test, y_test = load_raw(filepath, scale)

    print('x_train shape:', x_train.shape)  #(40091L, 150L,rows,cols,channels)
    print('y_train shape:', y_train.shape)  #(40091L,60L)
    print('x_test shape:',  x_test.shape)   #(16487L, 150L,rows,cols,channels)
    print('y_test shape',   y_test.shape)   #(16487L,60L)

    ################define the model############################
    model = get_model(layers_num,joint_num, MAX_LEN, stride_set,CLASS_NUM,reg,spp_numbers,spp_numbers_a,scale)
    if fine_tune:
        model.load_weights('model_scale3.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    model.summary()
    plot_model(model,to_file='model_scale'+scale+'.png')
    ## using multiple GPU model
    mgpu_model = keras.utils.multi_gpu_model(model,gpus=GPUS)
    mgpu_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    mgpu_model.summary()

    print('beginning to train the model....')
    mgpu_model.fit(x_train,y_train, batch_size= batch_size, epochs=epoch)

    print('Training duration (s) : ', time.time() - global_start_time)
    ################evaluate the model by using the test set##############################
    scores = mgpu_model.evaluate(x_test,y_test, batch_size = batch_size)
    print('Test score:', scores[0])
    print('Test accuracy:', scores[1])
    print('Training duration (s) : ', time.time() - global_start_time)

    return [model,scores[1]]

if __name__=='__main__':
    global_start_time = time.time()
    epochs = [60,80]
    test_score=[0,0,0]
    stride_set = [1,1,1]
    reg = regularizers.l2(LAMBDA1)
    batch_size = BATCH_SIZE
    max_len = MAX_LEN
    class_num = CLASS_NUM
    filepath = FILEPATH

    print('morphing the data into [samples, t, joint_num,3] format....')
    morph_data(filepath, max_len, scale='1', **MAT_NAMES)
    morph_data(filepath, max_len, scale='2', **MAT_NAMES)
    morph_data(filepath, max_len, scale='3', **MAT_NAMES)

    ###########################################################
    joint_num1  = 7*2
    spp_numbers1 = [4] # sum([ x**2 for x in spp_numbers1 ])*16 for the spp layer
    spp_numbers1_a = [7]
    layers_num1 = [joint_num1,4,8,16,100,100,sum([ x**2 for x in spp_numbers1 ])*16] ## 16,32,64,100##64,64,64,200 [input_layer,lstm1-lstm3,fc1_layer,fc2_layer] [joint_num1,50,50,50,100]
    model_scale1,test_score[0]=train_stream(layers_num1,batch_size,joint_num1,reg,filepath,stride_set,spp_numbers1,spp_numbers1_a,epoch = 40,fine_tune = False,scale='1')
    model_scale1.save('model_scale1.h5')    
    #########################################################
    joint_num2  = 12*2
    spp_numbers2 = [4]
    spp_numbers2_a = [12]
    layers_num2 = [joint_num2,8,16,32,100,100,sum([ x**2 for x in spp_numbers2 ])*32] ##128, 32,32,100,100##64,64,64,200 [input_layer,lstm1-lstm3,fc1_layer,fc2_layer] [joint_num2,80,80,80,160]
    model_scale2,test_score[1]=train_stream(layers_num2,batch_size,joint_num2,reg,filepath,stride_set,spp_numbers2,spp_numbers2_a,epoch=40,fine_tune = False,scale='2')
    model_scale2.save('model_scale2.h5')
    ##########################################################
    joint_num3  = 25*2
    spp_numbers3 = [4]
    spp_numbers3_a = [20]
    layers_num3 = [joint_num3,16,32,64,100,100,sum([ x**2 for x in spp_numbers3 ])*64] ##576, 32,64,64,200 [input_layer,lstm1-lstm3,fc1_layer,fc2_layer] [joint_num2,100,100,100,200]
    model_scale3,test_score[2]=train_stream(layers_num3,batch_size,joint_num3,reg,filepath,stride_set,spp_numbers3,spp_numbers3_a, epoch= 40,fine_tune = False,scale='3')
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
    x_train1, y_train1, x_test1, y_test1 = load_raw(filepath, '1')
    x_train2, y_train2, x_test2, y_test2 = load_raw(filepath, '2')
    x_train3, y_train3, x_test3, y_test3 = load_raw(filepath, '3')

    print('x_train shape:', x_train1.shape,x_train2.shape,x_train3.shape)  #(40091L, 150L,rows,cols,channels)
    print('y_train shape:', y_train1.shape,y_train2.shape,y_train3.shape)  #(40091L,60L)
    print('x_test shape:',  x_test1.shape,x_test2.shape,x_test3.shape)   #(16487L, 150L,rows,cols,channels)
    print('y_test shape',   y_test1.shape,y_test2.shape,y_test3.shape)   #(16487L,60L)

    if FUSION_MODE == 'feature':
        ### redirect the outputs####
        sub_model1 = Model(inputs = model_scale1.inputs, outputs = model_scale1.layers[-2].output,
                           name='stream_1')
        sub_model2 = Model(inputs = model_scale2.inputs, outputs = model_scale2.layers[-2].output,
                           name='stream_2')
        sub_model3 = Model(inputs = model_scale3.inputs, outputs = model_scale3.layers[-2].output,
                           name='stream_3')
        ## the three per-scale streams as they appear inside the fusion model
        stream_names = ['stream_1', 'stream_2', 'stream_3']

        ######### input to the three models
        fc_output1 = sub_model1(main_input1)
        fc_output2 = sub_model2(main_input2)
        fc_output3 = sub_model3(main_input3)

        fc_combined = keras.layers.concatenate([fc_output1,fc_output2,fc_output3])
        fc_combined_bn = BatchNormalization(name='BN_layer')(fc_combined)
        fc_softmax  = Dense(class_num,activation='softmax',kernel_regularizer = reg,bias_regularizer = reg,name = 'FC3')
        main_output = fc_softmax(Dropout(0.5)(fc_combined_bn))
    else:  # FUSION_MODE == 'prediction'
        ######### input to the three models
        fc_output1 = model_scale1(main_input1)
        fc_output2 = model_scale2(main_input2)
        fc_output3 = model_scale3(main_input3)

        main_output = keras.layers.average([fc_output1,fc_output2,fc_output3])
        ## here the streams are the renamed per-scale models themselves
        stream_names = ['model_1', 'model_2', 'model_3']
    
    model = Model(inputs=[main_input1,main_input2,main_input3], outputs=[main_output])
    model.summary()
    
    ###########################################################
    # Stage 1: train the fusion model with Adam (Sec. IV-B)
    print('jointly first fine-tune the whole fusion network with adam')
    adam_sgd = op.Adam(amsgrad=True)
    model.compile(loss='categorical_crossentropy', optimizer = adam_sgd,metrics=['accuracy'])
    ## using multiple GPU model
    mgpu_model = keras.utils.multi_gpu_model(model,gpus=GPUS)
    mgpu_model.compile(loss='categorical_crossentropy', optimizer = adam_sgd,metrics=['accuracy'])
    mgpu_model.summary()
    mgpu_model.fit([x_train1,x_train2,x_train3],y_train1, batch_size = batch_size, epochs=epochs[0])
    ## Checkpoint marking the Adam -> SGD stage boundary.
    model.save('full_fusion_model.h5')

    # Stage 2: continue fine-tuning with SGD (Sec. IV-B)
    print('jointly second fine-tune the whole fusion network with sgd')
    start_time = time.time()
    sgd = op.SGD(lr = 0.0001,momentum = 0.9)
    model.compile(loss='categorical_crossentropy', optimizer = sgd,metrics=['accuracy'])
    ## using multiple GPU model
    mgpu_model = keras.utils.multi_gpu_model(model,gpus=GPUS)
    mgpu_model.compile(loss='categorical_crossentropy', optimizer = sgd,metrics=['accuracy'])
    mgpu_model.summary()
    lr_reducer = LrReducer()
    tensorboard = TensorBoard()
    model_filepath="late_model_{epoch:02d}-{val_acc:.2f}.h5"
    checkpoint = ModelCheckpoint('./logs/'+ model_filepath, monitor = 'val_acc', save_weights_only=True)
    history = mgpu_model.fit([x_train1,x_train2,x_train3],y_train1, batch_size = batch_size, epochs=epochs[1],validation_split = 0.2, callbacks=[lr_reducer,tensorboard,checkpoint])

    print('Training duration (s) : ', time.time() - start_time)
    model.save('full_latefusion_model_final.h5')
    for i, stream_name in enumerate(stream_names, start=1):
        model.get_layer(stream_name).save('late_model_scale%d_ptrained.h5' % i)

    scores = mgpu_model.evaluate([x_test1,x_test2,x_test3],y_test1, batch_size = batch_size)
    print('Test score:', scores[0])
    print('Test accuracy:', scores[1])
    print('sub_model performance:',test_score)
    print('Training duration (s) : ', time.time() - global_start_time)
    
    plot_model(model,to_file='model.png')

    # Compare models' accuracy and loss per epoch.
    plot_training_history(history, 'latefusion_training.png')
