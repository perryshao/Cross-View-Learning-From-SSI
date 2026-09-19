'''MSNN fine-tuning on the Northwestern-UCLA Multiview 3D dataset.

    python conv-lstm-ucla-fusion-v2-ta-finetune.py -i 100 -lr 0.0001 -b 8 -d 1e-6

NOTE: consumes SSIs precomputed by `preprocessing/UCLA/compute_ssm_ucla_cv.py`;
it does not contain the learnable `MetricLayer` used by the NTU scripts.
'''
import time
import numpy as np
import matplotlib.pyplot as plt
import keras
import h5py
from keras.models import Model
from keras.layers import (BatchNormalization, Conv3D, Dense, Dropout, Input,
                          Lambda, LSTM, MaxPooling3D, Permute, wrappers)
from keras.utils import plot_model
from keras import regularizers
from keras import backend as K
import keras.optimizers as op
from keras_spp.spp.SpatialPyramidPooling import SpatialPyramidPooling
import argparse
     
def generate_train(filepath,batch_size,scale):
    while 1:
        file1 = h5py.File(filepath+'Trainset'+scale+'.h5','r')
        samples = file1['X_train'].shape[0]
        if samples%batch_size == 0:
            batch_times = (samples/batch_size)
        else:
            batch_times = (samples/batch_size)+1
        for cnt in range(batch_times-1): 
            X_train = file1['X_train'][cnt*batch_size:(cnt+1)*batch_size]
            Y_train = file1['Y_train'][cnt*batch_size:(cnt+1)*batch_size]
            yield (X_train,Y_train)
        cnt = cnt+1
        X_train = file1['X_train'][cnt*batch_size:]
        X_train_1 = np.concatenate((X_train[:,:1,:,:,:],X_train[:,:-1,:,:,:]),axis=1)
        Y_train = file1['Y_train'][cnt*batch_size:]
        x = X_train
        y = Y_train
        yield(x,y)
    file1.close()
def generate_test(filepath, batch_size, scale):
    while 1:
        file1 = h5py.File(filepath+'Testset'+scale+'.h5','r')
        samples = file1['X_test'].shape[0]
        if samples%batch_size == 0:
            batch_times = (samples/batch_size)
        else:
            batch_times = (samples/batch_size)+1
        for cnt in range(batch_times-1): 
            X_test = file1['X_test'][cnt*batch_size:(cnt+1)*batch_size]
            Y_test = file1['Y_test'][cnt*batch_size:(cnt+1)*batch_size]
            yield(X_test,Y_test)
        cnt = cnt+1
        X_test = file1['X_test'][cnt*batch_size:]
        Y_test = file1['Y_test'][cnt*batch_size:]
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
        file1 = h5py.File(filepath+'Trainset1'+'.h5','r')
        file2 = h5py.File(filepath+'Trainset2'+'.h5','r')
        file3 = h5py.File(filepath+'Trainset3'+'.h5','r')
        samples = file1['X_train'].shape[0]
        if samples%batch_size == 0:
            batch_times = (samples/batch_size)
        else:
            batch_times = (samples/batch_size)+1
        for cnt in range(batch_times-1): 
            X_train1 = file1['X_train'][cnt*batch_size:(cnt+1)*batch_size]
            Y_train1 = file1['Y_train'][cnt*batch_size:(cnt+1)*batch_size]
            X_train2 = file2['X_train'][cnt*batch_size:(cnt+1)*batch_size]
            Y_train2 = file2['Y_train'][cnt*batch_size:(cnt+1)*batch_size]
            X_train3 = file3['X_train'][cnt*batch_size:(cnt+1)*batch_size]
            Y_train3 = file3['Y_train'][cnt*batch_size:(cnt+1)*batch_size]
            x = [X_train1,X_train2,X_train3]
            y = Y_train1
            yield (x,y)
            
        cnt = cnt+1
        X_train1 = file1['X_train'][cnt*batch_size:]
        Y_train1 = file1['Y_train'][cnt*batch_size:]
        X_train2 = file2['X_train'][cnt*batch_size:]
        Y_train2 = file2['Y_train'][cnt*batch_size:]
        X_train3 = file3['X_train'][cnt*batch_size:]
        Y_train3 = file3['Y_train'][cnt*batch_size:]
        x = [X_train1,X_train2,X_train3]
        y = Y_train1
        yield(x,y)
    file1.close()
    file2.close()
    file3.close()
    
def generate_test_fusion(filepath, batch_size):
    while 1:
        file1 = h5py.File(filepath+'Testset1'+'.h5','r')
        file2 = h5py.File(filepath+'Testset2'+'.h5','r')
        file3 = h5py.File(filepath+'Testset3'+'.h5','r')
        samples = file1['X_test'].shape[0]
        if samples%batch_size == 0:
            batch_times = (samples/batch_size)
        else:
            batch_times = (samples/batch_size)+1
        for cnt in range(batch_times-1): 
            X_test1 = file1['X_test'][cnt*batch_size:(cnt+1)*batch_size]
            Y_test1 = file1['Y_test'][cnt*batch_size:(cnt+1)*batch_size]
            X_test2 = file2['X_test'][cnt*batch_size:(cnt+1)*batch_size]
            Y_test2 = file2['Y_test'][cnt*batch_size:(cnt+1)*batch_size]
            X_test3 = file3['X_test'][cnt*batch_size:(cnt+1)*batch_size]
            Y_test3 = file3['Y_test'][cnt*batch_size:(cnt+1)*batch_size]
            x = [X_test1,X_test2,X_test3]
            y = Y_test1
            yield (x,y)
            
        cnt = cnt+1
        X_test1 = file1['X_test'][cnt*batch_size:]
        Y_test1 = file1['Y_test'][cnt*batch_size:]
        X_test2 = file2['X_test'][cnt*batch_size:]
        Y_test2 = file2['Y_test'][cnt*batch_size:]
        X_test3 = file3['X_test'][cnt*batch_size:]
        Y_test3 = file3['Y_test'][cnt*batch_size:]
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
    assert len(shape) == 5  # only valid for 5D tensors
    shape= [input_shape[0],input_shape[1],input_shape[2]*input_shape[3]*input_shape[4]]
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
    return K.reshape(x,[-1,shape[1],K.cast(K.sqrt(shape[2]),dtype='int32'),K.cast(K.sqrt(shape[2]),dtype='int32'),1])
def unflattenConv_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3  # only valid for 3D tensors
    shape= [input_shape[0],input_shape[1],K.cast(K.sqrt(input_shape[2]),dtype='int32'),K.cast(K.sqrt(input_shape[2]),dtype='int32'),1]
    return tuple(shape)     


def get_model(layers_num,joint_num, max_len, stride_set,class_num, R3,spp_numbers,spp_numbers_a,scale,summary=False):
    
    ## parameter setting
    clip_length = 35
    main_layers = layers_num[0:5]  #layers [inputs,100,100,100] ## layer_num: [input_layer,lstm1-lstm3,fc1_layer,fc2_layer]
    fc1_layer = layers_num[5]
    
    ## define the input shape
    main_input = Input(shape=(max_len,main_layers[0],main_layers[0],1),dtype = 'float32', name = 'main_input'+'-'+scale)
    
    ########define the model###########
    c3d_out1 = Conv3D(main_layers[1],kernel_size=[3,3,3],strides = stride_set,activation = 'relu', kernel_regularizer=R3, bias_regularizer=R3,activity_regularizer=R3)(main_input)
    c3d_out2 = Conv3D(main_layers[2],kernel_size=[3,3,3],strides = stride_set,activation = 'relu', kernel_regularizer=R3, bias_regularizer=R3,activity_regularizer=R3)(c3d_out1)
    if scale == '1':
        c3d_out2 = MaxPooling3D(pool_size=(2, 1, 1),strides = (2,1,1))(c3d_out2)
        c3d_out3 = Conv3D(main_layers[3],kernel_size=[3,3,3],strides = stride_set,activation = 'relu',kernel_regularizer=R3, bias_regularizer=R3,activity_regularizer=R3)(c3d_out2)
        c3d_out3 = MaxPooling3D(pool_size=(2, 1, 1),strides = (2,1,1))(c3d_out3)
    if scale == '2':
        c3d_out2 = MaxPooling3D(pool_size=(2, 2, 2),strides = (2,2,2))(c3d_out2)
        c3d_out3 = Conv3D(main_layers[3],kernel_size=[3,3,3],strides = stride_set,activation = 'relu',kernel_regularizer=R3, bias_regularizer=R3,activity_regularizer=R3)(c3d_out2)
        c3d_out3 = MaxPooling3D(pool_size=(2, 1, 1),strides = (2,1,1))(c3d_out3)
    if scale == '3':
        c3d_out2 = MaxPooling3D(pool_size=(2, 2, 2),strides = (2,2,2))(c3d_out2)
        c3d_out3 = Conv3D(main_layers[3],kernel_size=[3,3,3],strides = stride_set,activation = 'relu',kernel_regularizer=R3, bias_regularizer=R3,activity_regularizer=R3)(c3d_out2)
        c3d_out3 = MaxPooling3D(pool_size=(2, 2, 2),strides = (2,2,2))(c3d_out3)
    
    spp_layer1 = wrappers.TimeDistributed(SpatialPyramidPooling(spp_numbers_a),name = 'TSPP1'+'-'+scale) 
    t_atten = spp_layer1(main_input)
    
    # t_atten = Lambda(flattenConv,flattenConv_output_shape)(main_input)
    fc1_ta = Dense(main_layers[4]/2, activation='relu',kernel_regularizer = R3,bias_regularizer = R3,name = 'FC1_ta_ucla')
    t_atten = fc1_ta(t_atten)
    t_atten = Permute((2, 1))(t_atten)
    fc2_ta = Dense(clip_length, activation='relu',kernel_regularizer = R3,bias_regularizer = R3,name = 'FC2_ta')
    t_atten = fc2_ta(t_atten)
    t_atten = Permute((2, 1))(t_atten)
    fc3_ta = Dense(layers_num[-1], activation='sigmoid',kernel_regularizer = R3,bias_regularizer = R3,name = 'FC3_ta')
    t_atten = fc3_ta(t_atten)

    
    spp_layer2 = wrappers.TimeDistributed(SpatialPyramidPooling(spp_numbers),name = 'TSPP2'+'-'+scale) 
    gru_input = spp_layer2(c3d_out3)
    
    # gru_input = Lambda(flattenConv,flattenConv_output_shape)(c3d_out3)
    gru_input = keras.layers.multiply([t_atten,gru_input])
    gru_output1 = LSTM(main_layers[4],return_sequences=True, dropout=0.5,recurrent_regularizer = R3,kernel_regularizer = R3,bias_regularizer = R3,
    name = 'lstm1'+'-'+scale)(gru_input)
    gru_output2 = LSTM(main_layers[4],return_sequences=True, dropout=0.5,recurrent_regularizer = R3,kernel_regularizer = R3,bias_regularizer = R3,
    name = 'lstm2'+'-'+scale)(gru_output1)
    

    fc1 = Dense(fc1_layer,activation='relu', kernel_regularizer = R3,bias_regularizer = R3,name = 'FC1'+'-'+scale)
    z_out = fc1(gru_output2)
    z_out_f = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]),name ='Lambda2'+'-'+scale)(z_out)
    
    
    fc2 = Dense(class_num,activation='softmax',kernel_regularizer = R3,bias_regularizer = R3,name = 'FC2'+'-'+scale)
    # # main_output = fc2(Dropout(0.5,name='Dropout2'+'-'+scale)(z_out_f))
    main_output = fc2(z_out_f)
    model = Model(inputs=[main_input], outputs=[main_output])
    if summary:
        print(model.summary())
    return model

def train_lstms(layers_num,batch_size,joint_num,Regs,filepath,stride_set,spp_numbers,spp_numbers_a,epoch,freeze = False,transfer =True,scale='1'):
    global_start_time = time.time()
    max_len = 150
    class_num = 10
    results=[]
    R1 = Regs[0]
    R2 = Regs[1]
    R3 = Regs[2]
    clip_length = 35
    
    print('> Loading data... ')
    
    file1 = h5py.File(filepath+'Trainset'+scale+'.h5','r')
    file2 = h5py.File(filepath+'Testset'+scale+'.h5','r')
    print('X_train shape:', file1['X_train'].shape)  #(40091L, 150L,rows,cols,channels)
    print('Y_train shape:', file1['Y_train'].shape)  #(40091L,60L)
    print('X_test shape:',  file2['X_test'].shape)   #(16487L, 150L,rows,cols,channels)
    print('Y_test shape',   file2['Y_test'].shape)   #(16487L,60L)
    if file1['X_train'].shape[0]%batch_size == 0:
        steps_train = file1['X_train'].shape[0]/batch_size
    else:
        steps_train = file1['X_train'].shape[0]/batch_size+1
    if file2['X_test'].shape[0]%batch_size == 0:
        steps_test = file2['X_test'].shape[0]/batch_size
    else:
        steps_test = file2['X_test'].shape[0]/batch_size+1
    
    file1.close()
    file2.close()
    print('> Data Loaded. Compiling...')
    
    ################define the model############################
    print("[Info] Reading model architecture at scale "+scale+"...")
    model = get_model(layers_num,joint_num, max_len, stride_set,class_num, R3,spp_numbers,spp_numbers_a,scale,summary=False)
    if transfer == True:
        print("[Info] Loading model weights...")
        model.load_weights('model_scale'+scale+'_ptrained.h5',by_name=True)
        print("[Info] Loading model weights -- DONE!")
    if freeze == True:
        model.layers[2].trainable = False
        model.layers[4].trainable = False
        model.layers[8].trainable = False
        model.layers[-4].trainable = False
        model.layers[-5].trainable = False
        
        # for l in model.layers[:-6]: # freeze the convlution layers, tune the lstm layers
                # l.trainable = False
    # sgd = op.SGD(lr=0.001, momentum=0.9, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer = 'adam',metrics=['accuracy'])
    model.summary()
    plot_model(model,to_file='model_scale'+scale+'.png')
    ## using multiple GPU model
    mgpu_model = keras.utils.multi_gpu_model(model,gpus=2)
    mgpu_model.compile(loss='categorical_crossentropy', optimizer = 'adam',metrics=['accuracy'])
    mgpu_model.summary()
    start_time = time.time()
    
    print('beginning to train the model....')
    
    history = mgpu_model.fit_generator(generate_train(filepath,batch_size,scale),steps_per_epoch=steps_train, epochs=epoch)


    average_time_per_epoch = (time.time() - start_time) / epoch
    print('Training duration (s) : ', time.time() - global_start_time)
    ################evaluate the model by using the test set##############################
    scores =mgpu_model.evaluate_generator(generate_test(filepath,batch_size,scale),steps=steps_test)
    print('Test score:', scores[0])
    print('Test accuracy:', scores[1])
    print('Training duration (s) : ', time.time() - global_start_time)
    
    plot_model(model,to_file='model.png')
    
    return [model,scores[1]]

def main(epochsFt, lrFt, batch_sizeFt,decayFt):
    
    global_start_time = time.time()
    epochs = [100,20]
    max_len = 150
    class_num = 10
    lamda1 = 0.01
    lamda2 = 0.001
    # lamda3 = 0.000005
    lamda3 = 0.000001
    test_score=[0,0,0]
    unit_length = 15
    results = []
    stride_set = [1,1,1]
    # R1 = regularizers.alpha_reg(lamda1,max_len)
    R1 = 0
    R2 = lamda2/max_len
    R3 = regularizers.l2(lamda3)
    Regs = [R1,R2,R3]
    filepath = '/home/data/UCLA_Multiview3D/'
    batch_size = 16
    

    ###########################################################
    joint_num1  = 7
    spp_numbers1 = [1,2,4] # sum([ x**2 for x in spp_numbers1 ])*16 for the spp layer
    spp_numbers1_a = [1,3,7]
    layers_num1 = [joint_num1,4,8,16,100,100,sum([ x**2 for x in spp_numbers1 ])*16] ## 16,32,64,100##64,64,64,200 [input_layer,lstm1-lstm3,fc1_layer,fc2_layer] [joint_num1,50,50,50,100]   
    # model_scale1,test_score[0] = train_lstms(layers_num1,batch_size,joint_num1,Regs,filepath,stride_set,spp_numbers1,spp_numbers1_a,epoch=20,freeze = False,transfer = True,scale='1')
    # model_scale1.save('uclamodel_scale1_v2_ta.h5')    
    #########################################################
    joint_num2  = 12
    spp_numbers2 = [1,2,4]
    spp_numbers2_a = [3,6,12]
    layers_num2 = [joint_num2,8,16,32,100,100,sum([ x**2 for x in spp_numbers2 ])*32] ##128, 32,32,100,100##64,64,64,200 [input_layer,lstm1-lstm3,fc1_layer,fc2_layer] [joint_num2,80,80,80,160]
    # model_scale2,test_score[1]=train_lstms(layers_num2,batch_size,joint_num2,Regs,filepath,stride_set,spp_numbers2,spp_numbers2_a,epoch=30,freeze = False,transfer = True,scale='2')
    # model_scale2.save('uclamodel_scale2_v2_ta.h5')
    ##########################################################
    joint_num3  = 20
    spp_numbers3 = [1,2,4]
    spp_numbers3_a = [5,10,20]
    layers_num3 = [joint_num3,16,32,64,100,100,sum([ x**2 for x in spp_numbers3 ])*64] ##576, 32,64,64,200 [input_layer,lstm1-lstm3,fc1_layer,fc2_layer] [joint_num2,100,100,100,200]
    model_scale3,test_score[2]=train_lstms(layers_num3,batch_size,joint_num3,Regs,filepath,stride_set,spp_numbers3,spp_numbers3_a,epoch=20,freeze = False,transfer = True, scale='3')
    model_scale3.save('uclamodel_scale3_v2_ta.h5')
    ###########################################################
    
    batch_size = batch_sizeFt  ########## batch_size for fine tune the whole network
    model_scale1 = get_model(layers_num1,joint_num1, max_len, stride_set,class_num, R3,spp_numbers1,spp_numbers1_a,scale = '1',summary=False)
    model_scale2 = get_model(layers_num2,joint_num2, max_len, stride_set,class_num, R3,spp_numbers2,spp_numbers2_a,scale = '2',summary=False)
    model_scale3 = get_model(layers_num3,joint_num3, max_len, stride_set,class_num, R3,spp_numbers3,spp_numbers3_a,scale = '3',summary=False)
    model_scale1.load_weights('uclamodel_scale1_v2_ta.h5')
    model_scale2.load_weights('uclamodel_scale2_v2_ta.h5')
    model_scale3.load_weights('uclamodel_scale3_v2_ta.h5')
    model_scale1.name ='model_1'
    model_scale2.name ='model_2'
    model_scale3.name ='model_3' # to avoid name conflicts of different models
    #####define the inputs
    main_input1= Input(shape=(max_len,joint_num1,joint_num1,1),dtype = 'float32')
    main_input2= Input(shape=(max_len,joint_num2,joint_num2,1),dtype = 'float32')	
    main_input3= Input(shape=(max_len,joint_num3,joint_num3,1),dtype = 'float32')
    ####determine the number of batches#########################
    file1 = h5py.File(filepath+'Trainset'+'1'+'.h5','r')
    file2 = h5py.File(filepath+'Testset'+'1'+'.h5','r')
    y_label = file1['Y_train']
    print('X_train shape:', file1['X_train'].shape)  #(40091L, 150L,rows,cols,channels)
    print('Y_train shape:', file1['Y_train'].shape)  #(40091L,60L)
    print('X_test shape:',  file2['X_test'].shape)   #(16487L, 150L,rows,cols,channels)
    print('Y_test shape',   file2['Y_test'].shape)   #(16487L,60L)
    if file1['X_train'].shape[0]%batch_size == 0:
        steps_train = file1['X_train'].shape[0]/batch_size
    else:
        steps_train = file1['X_train'].shape[0]/batch_size+1
    if file2['X_test'].shape[0]%batch_size == 0:
        steps_test = file2['X_test'].shape[0]/batch_size
    else:
        steps_test = file2['X_test'].shape[0]/batch_size+1

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
        fc_combined_bn = BatchNormalization(name='BN_layer')(fc_combined)
        fc3  = Dense(class_num,activation='softmax',kernel_regularizer = R3,bias_regularizer = R3,name = 'FC3')
        main_output = fc3(Dropout(0.5)(fc_combined_bn))
    else: #prediction fusion
        ######### input to the three models
        fc_output1 = model_scale1(main_input1)
        fc_output2 = model_scale2(main_input2)
        fc_output3 = model_scale3(main_input3)
    
        main_output = keras.layers.average([fc_output1,fc_output2,fc_output3])
    
    model = Model(inputs=[main_input1,main_input2,main_input3], outputs=[main_output])
    
    sgd = op.SGD(lr=lrFt, momentum=0.9, decay= decayFt)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
    model.summary()
    ## using multiple GPU model
    mgpu_model = keras.utils.multi_gpu_model(model,gpus=2) 
    mgpu_model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
    mgpu_model.summary()
    
    ###########################################################
    start_time = time.time()
    print('jointly fine-tune the whole fusion network')
    history = model.fit_generator(generate_train_fusion(filepath,batch_size),steps_per_epoch=steps_train, epochs=epochsFt)
    # history = mgpu_model.fit_generator(generate_train_fusion(filepath,batch_size),steps_per_epoch=steps_train, epochs=epochs[0])
    
    average_time_per_epoch = (time.time() - start_time) / epochsFt
    print('Training duration (s) : ', time.time() - global_start_time)
    model.save('full_fusion_model_ta.h5')
    # joblib.dump(history, 'ucla_history.pkl')
    
    scores = model.evaluate_generator(generate_test_fusion(filepath,batch_size),steps=steps_test)
    # scores = mgpu_model.evaluate_generator(generate_test_fusion(filepath,batch_size),steps=steps_test)
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
     
    
	
    #plot_results(predicted,Y_test,'predicted results')

    # Compare models' accuracy, loss and elapsed time per epoch.
    # history = joblib.load('ucla_history.pkl')
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
    plt.savefig("ucla_training_v4_sgd"+str(lrFt)+"_"+str(batch_sizeFt)+"_"+str(epochsFt)+".png")
    
    file = open('records.txt',"a+") 
    file.write('\n')
    file.write('ucla_training_v4_sgd'+str(lrFt)+'_'+str(batch_sizeFt)+'_'+str(epochsFt))
    file.write('\n')
    file.write('Training accuracy: '+str(history.history['acc'][-1]))
    file.write('\n')
    file.write('Test accuracy: '+str(scores[1]))  
    file.write('\n')
    file.close() 
    
    # predicted = model.predict([X_test,X_test,X_test_1])
    # print('predicted shape:',np.array(predicted).shape)  #(16488L,1L)
    # predicted = np.reshape(predicted, (predicted.size,)) #(16488L,)
if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--iterations", required=True,
        help="iterations for fine-tune cycle")
    ap.add_argument("-lr", "--learning_rate", required=True,
        help="learning rate for fine-tune cycle")
    ap.add_argument("-b", "--batch_size", required=True, 
        help="batch_size for fine-tune cycle")
    ap.add_argument("-d", "--decay", required=True, 
        help="decay coefficient for fine-tune cycle")
        
    epochs = ap.parse_args().iterations
    lr = ap.parse_args().learning_rate
    batchSize = ap.parse_args().batch_size
    decay = ap.parse_args().decay
    
    main(int(epochs), float(lr), int(batchSize),float(decay))
	
  
