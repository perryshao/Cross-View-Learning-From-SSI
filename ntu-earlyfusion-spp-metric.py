'''MSNN_early -- early-fusion Multi-Stream Neural Network on the NTU RGB+D dataset.

    python ntu-earlyfusion-spp-metric.py

The three per-scale streams are fused at the convolutional-feature level before
the LSTM sequence encoder, as described in Sec. III-B3 of the paper.
'''
import time
import h5py
import keras
import keras.optimizers as op
import numpy as np
from keras.models import Model
from keras.layers import (BatchNormalization, Conv3D, Dense, Input, Lambda,
                          LSTM, MaxPooling3D, Permute, wrappers)
from keras import regularizers
from keras import backend as K
from keras_spp.spp.SpatialPyramidPooling import SpatialPyramidPooling
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint, TensorBoard

from MetricLayer import MetricLayer
from data_utils import load_raw, morph_data, save_metric_matrix
from ssi_layers import (flattenSSM, flattenSSM_output_shape,
                        repeat_x_groupjoint, repeat_x_groupjoint_output_shape,
                        repeat_x_onejoint, repeat_x_onejoint_output_shape)
from viz_utils import LrReducer, plot_confusion_matrix, plot_training_history
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

## Per-scale MaxPooling3D configuration (cf. TABLE I).  Each entry is the
## (pool_after_conv2, pool_after_conv3) pair of pool sizes; strides match.
POOL_CONFIG = {
    '1': ((2, 1, 1), (2, 1, 1)),
    '2': ((2, 2, 2), (2, 1, 1)),
    '3': ((2, 2, 2), (2, 2, 2)),
}


def get_model(layers_num,joint_num, max_len, stride_set,class_num, R3,spp_numbers,spp_numbers_a,scale):
    
    ## parameter setting
    main_layers = layers_num  #layers [inputs,100,100,100] ## layer_num: [input_layer,lstm1-lstm3,fc1_layer,fc2_layer]
    conv_feats_dim = layers_num[-1]

    ## define the input shape
    main_input = Input(shape=(max_len,main_layers[0],3),dtype = 'float32', name = 'main_input'+'-'+scale)
    repeat_input1 = Lambda(repeat_x_onejoint,repeat_x_onejoint_output_shape)(main_input)
    repeat_input2 = Lambda(repeat_x_groupjoint,repeat_x_groupjoint_output_shape)(main_input)
    ########define the model###########
    ssm_input = keras.layers.subtract([repeat_input1,repeat_input2])
    ## Eq. (8) folds the metric weights L into the global l2 term ||W||_2, so
    ## the same regulariser is used here.  Pass `kernel_regularizer=None`
    ## instead to reproduce the unregularised L of the original runs.
    ssm_input = MetricLayer(joint_num,kernel_regularizer = R3,
                            name = 'metric_layer'+'-'+scale)(ssm_input)
    ssm_input = Lambda(flattenSSM,flattenSSM_output_shape)(ssm_input)

    pool2, pool3 = POOL_CONFIG[scale]
    c3d_out1 = Conv3D(main_layers[1],kernel_size=[3,3,3],strides = stride_set,activation = 'relu', kernel_regularizer=R3, bias_regularizer=R3,activity_regularizer=R3)(ssm_input)
    c3d_out2 = Conv3D(main_layers[2],kernel_size=[3,3,3],strides = stride_set,activation = 'relu', kernel_regularizer=R3, bias_regularizer=R3,activity_regularizer=R3)(c3d_out1)
    c3d_out2 = MaxPooling3D(pool_size=pool2, strides=pool2)(c3d_out2)
    c3d_out3 = Conv3D(main_layers[3],kernel_size=[3,3,3],strides = stride_set,activation = 'relu',kernel_regularizer=R3, bias_regularizer=R3,activity_regularizer=R3)(c3d_out2)
    c3d_out3 = MaxPooling3D(pool_size=pool3, strides=pool3)(c3d_out3)

    spp_layer1 = wrappers.TimeDistributed(SpatialPyramidPooling(spp_numbers_a),name = 'TSPP1'+'-'+scale)
    t_atten = spp_layer1(ssm_input)

    fc1_ta = Dense(main_layers[-2]//2, activation='relu',kernel_regularizer = R3,bias_regularizer = R3,name = 'FC1_ta')
    t_atten = fc1_ta(t_atten)
    t_atten = Permute((2, 1))(t_atten)
    fc2_ta = Dense(CLIP_LENGTH, activation='relu',kernel_regularizer = R3,bias_regularizer = R3,name = 'FC2_ta')
    t_atten = fc2_ta(t_atten)
    t_atten = Permute((2, 1))(t_atten) 
    fc3_ta = Dense(conv_feats_dim, activation='sigmoid',kernel_regularizer = R3,bias_regularizer = R3,name = 'FC3_ta')
    t_atten = fc3_ta(t_atten)
   
    spp_layer2 = wrappers.TimeDistributed(SpatialPyramidPooling(spp_numbers),name = 'TSPP2'+'-'+scale)
    lstm_input = spp_layer2(c3d_out3)

    lstm_input = keras.layers.multiply([t_atten,lstm_input])   
    model = Model(inputs=[main_input], outputs=[lstm_input])
    model.summary()
    return model

if __name__=='__main__':
    global_start_time = time.time()
    epochs = [60,60]
    max_len = MAX_LEN
    class_num = CLASS_NUM
    test_score=[0,0,0]
    stride_set = [1,1,1]
    R3 = regularizers.l2(LAMBDA1)
    lstm_layer_num = 100
    batch_size = BATCH_SIZE
    filepath = FILEPATH

    print('morphing the data into [samples, t, joint_num,3] format....')
    morph_data(filepath, max_len, scale='1')
    morph_data(filepath, max_len, scale='2')
    morph_data(filepath, max_len, scale='3')


    ###########################################################
    joint_num1  = 7*2 
    spp_numbers1 = [4] # sum([ x**2 for x in spp_numbers1 ])*16 for the spp layer
    spp_numbers1_a = [7]
    layers_num1 = [joint_num1,4,8,16,100,sum([ x**2 for x in spp_numbers1 ])*16] ## 16,32,64,100##64,64,64,200 [input_layer,lstm1-lstm3,fc1_layer,fc2_layer] [joint_num1,50,50,50,100]   
    #########################################################
    joint_num2  = 12*2
    spp_numbers2 = [4]
    spp_numbers2_a = [12]
    layers_num2 = [joint_num2,8,16,32,100,sum([ x**2 for x in spp_numbers2 ])*32] ##128, 32,32,100,100##64,64,64,200 [input_layer,lstm1-lstm3,fc1_layer,fc2_layer] [joint_num2,80,80,80,160]
    ##########################################################
    joint_num3  = 25*2
    spp_numbers3 = [4]
    spp_numbers3_a = [20]
    layers_num3 = [joint_num3,16,32,64,100,sum([ x**2 for x in spp_numbers3 ])*64] ##576, 32,64,64,200 [input_layer,lstm1-lstm3,fc1_layer,fc2_layer] [joint_num2,100,100,100,200]
    ###########################################################
    
    model_scale1 = get_model(layers_num1,joint_num1, max_len, stride_set,class_num, R3,spp_numbers1,spp_numbers1_a,scale = '1')
    model_scale2 = get_model(layers_num2,joint_num2, max_len, stride_set,class_num, R3,spp_numbers2,spp_numbers2_a,scale = '2')
    model_scale3 = get_model(layers_num3,joint_num3, max_len, stride_set,class_num, R3,spp_numbers3,spp_numbers3_a,scale = '3')
    model_scale1.name ='model_1'
    model_scale2.name ='model_2'
    model_scale3.name ='model_3' # to avoid name conflicts of different models
    #####define the inputs
    main_input1= Input(shape=(max_len,joint_num1,3),dtype = 'float32')
    main_input2= Input(shape=(max_len,joint_num2,3),dtype = 'float32')	
    main_input3= Input(shape=(max_len,joint_num3,3),dtype = 'float32')
    
    ####load the data of varying scales#########################
    x_train1, y_train1, x_test1, y_test1 = load_raw(filepath, '1')
    x_train2, y_train2, x_test2, y_test2 = load_raw(filepath, '2')
    x_train3, y_train3, x_test3, y_test3 = load_raw(filepath, '3')

    print('x_train shape:', x_train1.shape,x_train2.shape,x_train3.shape)  #(40091L, 150L,rows,cols,channels)
    print('y_train shape:', y_train1.shape,y_train2.shape,y_train3.shape)  #(40091L,60L)
    print('x_test shape:',  x_test1.shape,x_test2.shape,x_test3.shape)   #(16487L, 150L,rows,cols,channels)
    print('y_test shape',   y_test1.shape,y_test2.shape,y_test3.shape)   #(16487L,60L)

    ######### input to the three models
    feat_output1 = model_scale1(main_input1)
    feat_output2 = model_scale2(main_input2)
    feat_output3 = model_scale3(main_input3)
                 
    feat_combined = keras.layers.concatenate([feat_output1,feat_output2,feat_output3])       
    feat_combined_bn = BatchNormalization(name='BN_layer')(feat_combined)
    lstm_output1 = LSTM(lstm_layer_num,return_sequences=True, dropout=0.5,recurrent_regularizer = R3,kernel_regularizer = R3,bias_regularizer = R3,
    name = 'lstm1')(feat_combined_bn)
    lstm_output2 = LSTM(lstm_layer_num,return_sequences=True, dropout=0.5,recurrent_regularizer = R3,kernel_regularizer = R3,bias_regularizer = R3,
    name = 'lstm2')(lstm_output1)
    z_out_f = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]),name ='Lambda2')(lstm_output2) 
    fc_softmax = Dense(class_num,activation='softmax',kernel_regularizer = R3,bias_regularizer = R3,name = 'FC2')
    main_output = fc_softmax(z_out_f)

    model = Model(inputs=[main_input1,main_input2,main_input3], outputs=[main_output])
    model.summary()
    ## To inspect a stream's feature maps, e.g.:
    ##   from viz_utils import visualize_layer
    ##   visualize_layer(model_scale3, 'max_pooling3d_9', x_train3[11:12], 24)
    adam_sgd = op.Adam(amsgrad=True)
    sgd = op.SGD(lr = 0.0001,momentum = 0.9)
    ###########################################################
    # Stage 1: train the fusion model with Adam (Sec. IV-B)
    print('jointly first fine-tune the whole fusion network with adam')
    model.compile(loss='categorical_crossentropy', optimizer = adam_sgd, metrics=['accuracy'])
    ## using multiple GPU model
    mgpu_model = keras.utils.multi_gpu_model(model,gpus=GPUS)
    mgpu_model.compile(loss='categorical_crossentropy', optimizer = adam_sgd, metrics=['accuracy'])
    mgpu_model.summary()
    mgpu_model.fit([x_train1,x_train2,x_train3],y_train1, batch_size = batch_size, epochs=epochs[0]) #epochs=epochs[0] for adam
    ## Checkpoint marking the Adam -> SGD stage boundary.
    model.save('full_earlyfusion_model.h5')

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
    checkpoint = ModelCheckpoint('./logs/'+ model_filepath, monitor = 'val_acc', save_weights_only=True)
    history = mgpu_model.fit([x_train1,x_train2,x_train3],y_train1, batch_size = batch_size, epochs=epochs[1],validation_split = 0.2, callbacks=[lr_reducer,tensorboard,checkpoint]) #epochs=epochs[0] for adam


    print('Training duration (s) : ', time.time() - start_time)
    scores =mgpu_model.evaluate([x_test1,x_test2,x_test3],y_test1, batch_size = batch_size)
    print('Test score:', scores[0])
    print('Test accuracy:', scores[1])
    print('sub_model performance:',test_score)
    print('Training duration (s) : ', time.time() - global_start_time)

    model.save('full_earlyfusion_final.h5')
    ## Save each per-scale stream and dump its learned Mahalanobis transform L.
    for scale in ('1', '2', '3'):
        stream = model.get_layer('model_' + scale)
        stream.save('early_model_scale' + scale + '_ptrained.h5')
        weights = stream.get_layer('metric_layer' + '-' + scale).get_weights()
        save_metric_matrix(weights[0], scale=scale)

    y_pred = mgpu_model.predict([x_test1,x_test2,x_test3])
    matrix = confusion_matrix(y_test1.argmax(axis=1), y_pred.argmax(axis=1))
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    ## plot confusion matrix using matplotlib
    class_names = ['drink water', 'eat meal/snack', 'brushing teeth', 'brushing hair', 'drop', 'pickup', 'throw', 'sitting down',
    'standing up','clapping', 'reading', 'writing', 'tear up paper', 'wear jacket', 'take off jacket',
    'wear a shoe','take off a shoe','wear on glasses','take off glasses','put on a hat/cap',
    'take off a hat/cap','cheer up','hand waving','kicking something','put something inside pocket',
    'hopping (one foot jumping)','jump up','make a phone call','playing with phone/tablet','typing on a keyboard',
    'pointing to something with finger','taking a selfie','check time (from watch)','rub two hands together',
    'nod head/bow','shake head','wipe face','salute','put the palms together','cross hands in front (say stop)',
    'sneeze/cough','staggering','falling','touch head (headache)','touch chest (stomachache/heart pain)',
    'touch back (backache)','touch neck (neckache)','nausea or vomiting condition','use a fan (with hand or paper)/feeling warm',
    'punching/slapping other person','kicking other person','pushing other person','pat on back of other person',
    'point finger at the other person','hugging other person','giving something to other person','touch other person''s pocket',
    'handshaking','walking towards each other','walking apart from each other']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
    plt.savefig("confusion_matrix.png")

    ## save the confusion matrix as .h5 file so that the matrix can be plot with Matlab
    with h5py.File('matrix.h5','w') as file1:
        file1.create_dataset('matrix', data = matrix)

    # Compare models' accuracy and loss per epoch.
    plot_training_history(history, 'ntu_training.png')
