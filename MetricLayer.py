from keras import backend as K
from keras.engine.topology import Layer
from keras import regularizers

class MetricLayer(Layer):

    def __init__(self, output_dim, kernel_regularizer=None, **kwargs):
        self.output_dim  = output_dim
        super(MetricLayer, self).__init__(**kwargs)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      # shape=(input_shape[1],input_shape[-1],input_shape[-1]),
                                      shape=(input_shape[-1],input_shape[-1]),
                                      initializer='uniform',
                                      regularizer=self.kernel_regularizer, 
                                      trainable=True)
        super(MetricLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, input): 

        ## Tx3x3 weight Matrix
        # y1 = list()
        # for t in range(input.shape[1]):
        #     ssm = K.sqrt(K.dot(K.square(K.dot(input[:,t,:,:], self.kernel[t,:,:])),K.ones(shape=(3, 1)))+K.epsilon())

        #     ssm_var = K.var(ssm, axis=1, keepdims=True)
        #     ssm = (ssm-K.repeat_elements(K.mean(ssm,axis=1,keepdims=True),ssm.shape[1],axis=1))/K.repeat_elements(K.sqrt(ssm_var+K.epsilon()),ssm.shape[1],axis=1)

        #     # ssm = (ssm - K.min(ssm,axis=1,keepdims=True))/(K.max(ssm,axis=1,keepdims=True) - K.min(ssm,axis=1,keepdims=True)+K.epsilon())
        #     y1.append(ssm)
        # x = K.stack(y1,axis=1)

        ## 3x3 weight matrix
        x = K.sum(K.square(K.dot(input, self.kernel)), axis =-1, keepdims = True)
        x_st = K.reshape(x,[-1,x.shape[1]*x.shape[2],x.shape[-1]])
        x_st_max = K.reshape(K.max(x_st,axis=-2,keepdims=True),[-1,1,1,x.shape[-1]])
        x_st_min = K.reshape(K.min(x_st,axis=-2,keepdims=True),[-1,1,1,x.shape[-1]])
        x = (x - x_st_min)/(x_st_max - x_st_min+K.epsilon())

        return x

    def compute_output_shape(self, input_shape):
        # shape = list(input_shape)
        # output_shape = K.reshape(shape,[shape[0],self.output_dim,self.output_dim,-1])
        return (input_shape[0], input_shape[1], input_shape[2], 1)