
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd


# In[2]:

# This function is called from Main and expects train and test values for x and y
def load_ag_data(authors = None, docID = None): 
    
    import data
    train = data.getCharAuthorData(authors, docID) #Pass it to data and it returns a data frame
    train = train.dropna()

    labels = [] #
    texts = []
    size = []
    authorList = train.author_id.unique()
    for auth in authorList:
        current = train.loc[train['author_id'] == auth]
        size.append(current.shape[0])
        print("Author: %5s  Size: %5s" % (auth, current.shape[0]))
    print("Min: %s" % (min(size)))
    print("Max: %s" % (max(size)))

    authorList = authorList.tolist()

    for auth in authorList:
        current = train.loc[train['author_id'] == auth]
        samples = min(size)
        current = current.sample(n = samples)
        textlist = current.doc_content.tolist()
        texts = texts + textlist
        labels = labels + [authorList.index(author_id) for author_id in current.author_id.tolist()]
    labels_index = {}
    labels_index[0] = 0
    for i, auth in enumerate(authorList):
        labels_index[i] = auth

    del train
    from keras.utils.np_utils import to_categorical
    labels = to_categorical(labels)
    
    print('Authors %s.' % (str(authorList)))
    print('Found %s texts.' % len(texts))
    print('Found %s labels.' % len(labels))
    
    from sklearn.model_selection import train_test_split
    trainX, valX, trainY, valY = train_test_split(texts, labels, test_size= 0.2)
    
    
    # return (texts, labels, labels_index, samples)


    return ((trainX, trainY), (valX, valY))


def load_doc_data(authors = None, docID = None):
    import data
    test = data.getCharDocData(authors, docID) #Pass it to data and it returns a data frame
    test = test.dropna()
    
    labels = []
    texts = []
    for index, row in test.iterrows():
        labels.append(authors.index(row.author_id))
        texts.append(row.doc_content)

    del test # Garbage collection

    print('Found %s texts.' % len(texts))
    return (texts, labels)
    

def mini_batch_generator(x, y, vocab, vocab_size, vocab_check, maxlen,
                         batch_size=128):

    for i in xrange(0, len(x), batch_size):
        x_sample = x[i:i + batch_size]
        y_sample = y[i:i + batch_size]

        input_data = encode_data(x_sample, maxlen, vocab, vocab_size,
                                 vocab_check)

        yield (input_data, y_sample)

def shuffle_matrix(x, y):
    stacked = np.hstack((np.matrix(x).T, y))
    np.random.shuffle(stacked)
    xi = np.array(stacked[:, 0]).flatten()
    yi = np.array(stacked[:, 1:])

    return xi, yi

def encode_data(x, maxlen, vocab, vocab_size, check):
    #Iterate over the loaded data and create a matrix of size maxlen x vocabsize
    #In this case that will be 1014x69. This is then placed in a 3D matrix of size
    #data_samples x maxlen x vocab_size. Each character is encoded into a one-hot
    #array. Chars not in the vocab are encoded into an all zero vector.

    input_data = np.zeros((len(x), maxlen, vocab_size))
    for dix, sent in enumerate(x):
        counter = 0
        sent_array = np.zeros((maxlen, vocab_size))
        chars = list(sent.replace(' ', ''))
        for c in chars:
            if counter >= maxlen:
                pass
            else:
                char_array = np.zeros(vocab_size, dtype=np.int)
                if c in check:
                    ix = vocab[c]
                    char_array[ix] = 1
                sent_array[counter, :] = char_array
                counter += 1
        input_data[dix, :, :] = sent_array

    return input_data


# In[12]:

def create_vocab_set():
    #This is a Unicode Character set
    import string
    unicode_characters = [];
    #initially 1280
    for k in range(0, 500):
        unicode_characters.append(unichr(k))
        
    #or k in range(1024, 1280):
        #unicode_characters.append(unichr(k))


    alphabet = unicode_characters
    vocab_size = len(alphabet)
    check = set(alphabet)
    vocab = {}
    reverse_vocab = {}
    for ix, t in enumerate(alphabet):
        vocab[t] = ix
        reverse_vocab[ix] = t


    return vocab, reverse_vocab, vocab_size, check


# In[16]:

# (vocab, reverse_vocab, vocab_size, check) = create_vocab_set()

