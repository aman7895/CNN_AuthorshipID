
# coding: utf-8

# In[4]:

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution1D, MaxPooling1D


# In[3]:


def model(classes, filter_kernels, dense_outputs, maxlen, vocab_size, nb_filter):
    #Define what the input shape looks like
    
    model = Sequential()
    
    model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[0],
                         border_mode='valid', activation='relu',
                         input_shape=(maxlen, vocab_size)))
    model.add(MaxPooling1D(pool_length=3))
    
    """
    model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[1],
                         border_mode='valid', activation='relu',
                         input_shape=(maxlen, vocab_size)))
    model.add(MaxPooling1D(pool_length=3))
    
    model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[2],
                         border_mode='valid', activation='relu',
                         input_shape=(maxlen, vocab_size)))
    
    model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[3],
                         border_mode='valid', activation='relu',
                         input_shape=(maxlen, vocab_size)))
    
    model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[4],
                         border_mode='valid', activation='relu',
                         input_shape=(maxlen, vocab_size)))
    
    model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[5],
                         border_mode='valid', activation='relu',
                         input_shape=(maxlen, vocab_size)))
    model.add(MaxPooling1D(pool_length=3))
    """
    
    model.add(Flatten())
    
    """
    model.add(Dense(dense_outputs, activation='relu'))
    model.add(Dropout(0.5))
    """
    
    model.add(Dense(dense_outputs, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(classes, activation='softmax', name='output'))
    
    sgd = SGD(lr=0.01, momentum=0.9, nesterov= True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])

    return model


# In[ ]:



