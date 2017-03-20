
# coding: utf-8

# In[4]:

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution1D, MaxPooling1D


# In[3]:


def build_model(classes, filter_kernels, dense_outputs, maxlen, vocab_size, nb_filter):
    #Define what the input shape looks like
    
    model = Sequential()
    
    model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[0],
                            border_mode='valid', activation='relu',
                            input_shape=(maxlen, vocab_size),
                            name = 'conv1'))
    model.add(MaxPooling1D(pool_length=3, name = 'maxpool1'))
    
    """
    model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[1],
                         border_mode='valid', activation='relu',
                         input_shape=(maxlen, vocab_size), name = 'conv2'))
    model.add(MaxPooling1D(pool_length=3, name = 'maxpool2))
    
    model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[2],
                         border_mode='valid', activation='relu',
                         input_shape=(maxlen, vocab_size), name = 'conv3'))
    
    model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[3],
                         border_mode='valid', activation='relu',
                         input_shape=(maxlen, vocab_size), name = 'conv4'))
    
    model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[4],
                         border_mode='valid', activation='relu',
                         input_shape=(maxlen, vocab_size), name = 'conv5'))
    
    model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[5],
                         border_mode='valid', activation='relu',
                         input_shape=(maxlen, vocab_size), name = 'conv6'))
    model.add(MaxPooling1D(pool_length=3, name = 'maxpool3'))
    """
    
    model.add(Flatten())
    
    model.add(Dense(dense_outputs, activation='relu', name = 'dense1'))
    model.add(Dropout(0.5, name = 'dropout1'))
    
    """
    model.add(Dense(dense_outputs, activation='relu', name = 'desne2'))
    model.add(Dropout(0.5, name = dropout2))
    """
    
    model.add(Dense(classes, activation='softmax', name='output'))
    
    sgd = SGD(lr=0.01, momentum=0.9, nesterov= True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])
    
    filepath = 'params/crepe_model_weights.h5'

    return (model, sgd, filepath)


def build_feature_model(classes, filter_kernels, dense_outputs, maxlen, vocab_size, nb_filter, sgd, filepath):
    #Define what the input shape looks like
    
    model = Sequential()
    
    model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[0],
                            border_mode='valid', activation='relu',
                            input_shape=(maxlen, vocab_size),
                            name = 'conv1', trainable = False))
    model.add(MaxPooling1D(pool_length=3, name = 'maxpool1', trainable = False))
    
    """
    model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[1],
                         border_mode='valid', activation='relu',
                         input_shape=(maxlen, vocab_size), name = 'conv2'))
    model.add(MaxPooling1D(pool_length=3, name = 'maxpool2))
    
    model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[2],
                         border_mode='valid', activation='relu',
                         input_shape=(maxlen, vocab_size), name = 'conv3'))
    
    model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[3],
                         border_mode='valid', activation='relu',
                         input_shape=(maxlen, vocab_size), name = 'conv4'))
    
    model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[4],
                         border_mode='valid', activation='relu',
                         input_shape=(maxlen, vocab_size), name = 'conv5'))
    
    model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[5],
                         border_mode='valid', activation='relu',
                         input_shape=(maxlen, vocab_size), name = 'conv6'))
    model.add(MaxPooling1D(pool_length=3, name = 'maxpool3'))
    """
    
    model.add(Flatten())
    
    model.add(Dense(dense_outputs, activation='relu', name = 'dense1', trainable = False))
    model.add(Dropout(0.5, name = 'dropout1', trainable = False))
    
    """
    model.add(Dense(dense_outputs, activation='relu', name = 'desne2'))
    model.add(Dropout(0.5, name = dropout2))
    """
    
    model.add(Dense(classes, activation='softmax', name='output', trainable = False))
    
    model.load_weights(filepath)
    
    model.pop()  #pop dense output
    
    model.pop()  #pop dropout
    
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])
    
    return (model)


# In[ ]:



