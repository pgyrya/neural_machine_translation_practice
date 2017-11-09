# -*- Encoding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

#from sklearn                  import preprocessing

#import backports 
#from backports import weakref
#import tensorflow
#from keras.preprocessing.text import text_to_word_sequence
import keras
from keras.models             import Model
from keras.layers             import Input, Dense #, Permute, Multiply, LSTM, Concatenate, Flatten
#from keras.layers.core        import RepeatVector
from keras.callbacks          import ModelCheckpoint  

#import seq2seq
#from seq2seq.models           import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq


###############################################################################
# Step 1 - import data 
# Initially, subset to a small batch to see whether the process 
# I intend to implement works -e.g. working with example of numbers only
###############################################################################
#df = pd.read_csv("test.csv")
#df = pd.read_csv("test2.csv")
df0 = pd.read_csv("en_train.csv", dtype = {'sentence_id': np.int32,	
                                          'token_id'   : np.int32,	
                                          'class' : str,	
                                          'before': str,	
                                          'after' : str})
total_number_of_input_phrases           = df0.shape[0]
print('Imported ', total_number_of_input_phrases, 'records')

#review imported dataset
#print(df0.loc[9:28,['class', 'before','after']])
#print(df0.loc[number_of_input_phrases-10:number_of_input_phrases-1,['class', 'before','after']])

#Subset the original data frame to areas of interest for practice
#df = df0[df0['class'] =='CARDINAL'].reset_index() #'DIGIT', 'DECIMAL')
selected_types_set = {'CARDINAL'}
#selected_types_set = {'DIGIT','CARDINAL','ORDINAL','DECIMAL', 'FRACTION', 'MONEY', 'MEASURE', 'DATE', 'TELEPHONE', 'TIME'} 
#not_selected_types_set = {'PLAIN', 'PUNCT'} #more to be added
type_match = lambda type: type in selected_types_set
df1 = df0[df0['class'].apply(type_match)==True].reset_index(drop=True)

# Next, focus on cardinals written with at most 10 symbols
df  = df1[df1['before'].apply(len)<=10].reset_index(drop=True)

#print sample
#print(df.loc[9:28,['class', 'before','after']])





number_of_input_phrases              = df.shape[0]
print('Selected %i records out of %i for modeling.'%(number_of_input_phrases,total_number_of_input_phrases))

#df.head()
#df.shape

#Split input data into development, validation and testing data frames
np.random.seed(217)
df['r'] = [random.uniform(0,1) for _ in df.index]
df.to_csv('en_train_cardinals.csv')
#print(df['r'])

df_train= df[ df['r'] <  0.5                     ].reset_index(drop=True)
df_val  = df[(df['r'] >= 0.5 ) & (df['r'] < 0.75)].reset_index(drop=True)
df_test = df[ df['r'] >= 0.75                    ].reset_index(drop=True)


data_size_train = df_train.shape[0]
data_size_val   = df_val.shape[0]
data_size_test  = df_test.shape[0]

print('Split selected records into ',
      '\n - training sample (%i records), '%data_size_train,
      '\n - validation sample (%i records) used to select best model candidates,'%data_size_val,
      '\n - testing sample (%i records) used to review model quality'%data_size_test)

#print(df['before'].apply(len))

###############################################################################
# Step 2   - Transform imported data into vector form
#      2.1 - Identify tokens that are part of the input and output vocabulary
#      ...
###############################################################################
from collections import Counter
from operator import itemgetter
DEFAULT_TOKEN = '[?]'

str_to_list_input  = lambda s: list(str(s).lower())
str_to_list_output = lambda s: str(s).lower().split()  + ['\n']

count_input_tokens = lambda s: len(str_to_list_input(s))
count_output_tokens= lambda s: len(str_to_list_output(s))

# test functionality
str_to_list_input ('Test Test 1 2 3')
str_to_list_output ('Test Test 1 2 3')

#input_list_of_lists = df['before'].apply(str_to_list_input)
#output_list_of_lists= df['after'].apply(str_to_list_output)
# TODO - consider appending start and end tokens to beginning and the end

input_symbols_counter = Counter([i for slist in df['before'].apply(str_to_list_input) 
                                 for i in slist])
output_symbols_counter= Counter([i for slist in df['after'].apply(str_to_list_output) 
                                 for i in slist]) 

#len(input_symbols_counter)
#len(output_symbols_counter)

def top(input_dict, top_N = 200):
    appearances_count = dict(sorted(
            input_dict.items(), key=itemgetter(1), reverse = True))
       
    #tokenize first top_N items as separate items, the rest as single item
    index = dict([(key, min(i,top_N)) for i, key in enumerate(appearances_count.keys())])
    
    #setup reverse lookup, with default symbol replacing non-common tokens
    reverse_index = dict([(i, key) for key, i in index.items() if i<top_N])
    reverse_index[top_N] = DEFAULT_TOKEN
    
    return index, reverse_index, appearances_count 

input_token_index, reverse_input_token_index, input_token_appearances = top(
        input_symbols_counter, 100)

output_token_index, reverse_output_token_index, output_token_appearances = top(
        output_symbols_counter, 1000)

print(output_token_appearances)
#input_token_index
#len(input_token_index)
#input_token_appearances

num_distinct_input_tokens  = len(reverse_input_token_index)
num_distinct_output_tokens = len(reverse_output_token_index)

print ('Number of distinct input tokens used:' , num_distinct_input_tokens )
print ('%d symbols represented via default input token.' % (len(input_token_index) - num_distinct_input_tokens + 1) )

print ('Number of distinct output tokens used:', num_distinct_output_tokens)
print ('%d symbols represented via default input token.' % (len(output_token_index) - num_distinct_output_tokens + 1) )

#print(input_token_appearances)

###############################################################################
# Step 2   - Transform imported data into vector form
#      ...
#      2.2 - Vectorize input via one-hot encoding into a Numpy array, preparing
#            to feed input symbols from before column into seq-to-seq model
#            Also vectorize output, prepating to feed translations as outputs
###############################################################################

max_tokens_per_input_sentence    = 10 #25 
max_tokens_per_output_sentence   = 10 #25 #100

# initialize encoder input data as generator of 2D NumPy array with dimensions:
# 1) maximum sentence length for encoder
# 2) number of encoder tokens (origin language)

# initialize decoder input data as generator of 2D NumPy array with dimensions:
# 1) maximum sentence length for decoder
# 2) number of decoder tokens (destination language)

input_shape = (max_tokens_per_input_sentence, num_distinct_input_tokens)
output_shape= (max_tokens_per_output_sentence, num_distinct_output_tokens)


#encoder_input_data = np.zeros(input_shape, dtype='bool')
#decoder_target_data = np.zeros(output_shape, dtype='bool')


import random

print('Shape of the encoder input data is: (%i,%i,%i)'   %(number_of_input_phrases, input_shape[0] ,input_shape[1] ))
print('Shape of the decoder target data is: (%i,%i,%i) ' %(number_of_input_phrases, output_shape[0],output_shape[1]))
print('Training and validation split are: (%i,%i)'       %(data_size_train,         data_size_val))

# fill in encoder input  data representing characters via one-hot encoding
const_one = lambda seq: 1
def lookup_and_transform(
        dataset, 
        index_range, 
        verbose = False, 
        weight_function = const_one #could use const_one or len, for example
        ):
    index_list = list(index_range)
    number_of_samples    = len(index_range)    
    encoder_input        = np.zeros((number_of_samples,) + input_shape  , dtype='float32')
    decoder_target       = np.zeros((number_of_samples,) + output_shape , dtype='float32')
    decoder_time_weights = np.zeros((number_of_samples, output_shape[0]), dtype='float32')   #zero weight for decoder would mean disregarding any mismatches for this time point
    #print(encoder_input.shape)

    input_sequences = dataset['before'][index_list].apply(str_to_list_input ).tolist()    
    target_sequences= dataset['after' ][index_list].apply(str_to_list_output).tolist()
    #print (dataset.head(10))
    
    if verbose==True:
        for input_sequence, target_sequence in zip(input_sequences, target_sequences):
            print('Input sequence :', input_sequence, 
                  '\n Corresponding target sequence:', target_sequence)
        
    for sample in range(number_of_samples):
        for t, input_token in enumerate(input_sequences[sample]):
            #if verbose==True:
            #    print ('Processing input token ', input_token,
            #           'with index ', input_token_index[input_token])
            if t < max_tokens_per_input_sentence: 
                encoder_input[sample,t, input_token_index[input_token]] = 1.
                
        for t, output_token in enumerate(target_sequences[sample]):
            #if verbose==True:
            #    print ('Processing output token ', output_token,
            #           'with index ', output_token_index[output_token])
            if t < max_tokens_per_output_sentence:
                decoder_target[sample,t, output_token_index[output_token]] = 1. 
                decoder_time_weights[sample, t] = weight_function(target_sequences[sample]) #previously set to 1
                if (output_token.lower() == 'nan'):  # Write warning message
                    print('Warning: NaN output token detected in subsample, row #', sample, 
                          'In the sentence:', ' '.join(target_sequences[sample]),
                          'Index range requested: (%i,%i)'% (index_range[0], index_range[-1]))
            
    return encoder_input, decoder_target, decoder_time_weights

#encoder_input, decoder_target, decoder_time_weights = lookup_and_transform(df_train,range(0,1), True)
#encoder_input.sum()
#encoder_input
#lookup_and_transform(df_train,range(0,64), True)
#lookup_and_transform(df_train,range(0,64), True)[0].shape
    
#lookup_and_transform(df_train,range(0,64))
#list([1,2,3])
#list(range(1,3+1))
#print(df_train.loc[9])
#print (reverse_input_token_index)    
#len(range(1,3+1))

# These generators are expected to loop over data indefinitely, generating batches
# of data for model development and validation    
def training_data_generator(batch_size = 64, dataset = df_train, verbose = False):
    batch_index = 0    
    batches_fitting_in_train_data = int(dataset.shape[0]/ batch_size)
    while True:        
        #print ('Generating training data with index from %i to %i'%(batch_size *  batch_index,batch_size * (batch_index + 1)-1))
        yield lookup_and_transform(dataset, 
                                   range(batch_size * batch_index, batch_size * (batch_index + 1)),
                                   verbose)
        batch_index += 1
        if batch_index >= batches_fitting_in_train_data: batch_index = 0 #restart generator


def validation_data_generator(batch_size = 64, dataset = df_val):
    batch_index_val = 0    
    batches_fitting_in_val_data = int(dataset.shape[0]/ batch_size)
    while True: 
        #print ('Generating validation data with index from %i to %i'%(batch_size *  batch_index_val,batch_size * (batch_index_val + 1)-1))
        yield lookup_and_transform(dataset, 
                                   range(batch_size *  batch_index_val, batch_size * (batch_index_val + 1)))
        batch_index_val += 1
        if batch_index_val >= batches_fitting_in_val_data: batch_index_val = 0 #restart generator
    

###############################################################################
# Step 3 - Set up recurrent neural network architecture
###############################################################################
#dims = [32]
#dims = [num_distinct_input_tokens] # 67
dims = [64]

#sample models offered by github documentation:
#model = SimpleSeq2Seq(input_dim=5, hidden_dim=10, output_length=8, output_dim=8)
#model = Seq2Seq(batch_input_shape=(16, 7, 5), hidden_dim=10, output_length=8, output_dim=20, depth=4, peek=True)
#model = AttentionSeq2Seq(input_dim=5, input_length=7, hidden_dim=10, output_length=8, output_dim=20, depth=4)
from recurrentshop import LSTMCell, RecurrentSequential
from seq2seq.cells import LSTMDecoderCell, AttentionDecoderCell
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, TimeDistributed, Bidirectional, Input

def AttentionSeq2Seq(output_dim, 
                     output_length, 
                     batch_input_shape= None,
                     batch_size       = None, 
                     input_shape      = None, 
                     input_length     = None,
                     input_dim        = None, 
                     hidden_dim       = None, 
                     depth            = 1,
                     bidirectional    = True, 
                     unroll           = False, 
                     stateful         = False, 
                     dropout          = 0.0,
                     ):
    '''
    This is an attention Seq2seq model based on [3].
    Here, there is a soft allignment between the input and output sequence elements.
    A bidirection encoder is used by default. There is no hidden state transfer in this
    model.

    The  math:

    Encoder:
    X = Input Sequence of length m.
    H = Bidirection_LSTM(X); Note that here the LSTM has return_sequences = True,
    so H is a sequence of vectors of length m.

    Decoder:
    y(i) = LSTM(s(i-1), y(i-1), v(i)); Where s is the hidden state of the LSTM (h and c)
    and v (called the context vector) is a weighted sum over H:

    v(i) =  sigma(j = 0 to m-1)  alpha(i, j) * H(j)

    The weight alpha[i, j] for each hj is computed as follows:
    energy = a(s(i-1), H(j))
    alpha = softmax(energy)
    Where a is a feed forward network.

    '''

    if isinstance(depth, int): depth = (depth, depth)
    if batch_input_shape:      shape = batch_input_shape
    elif input_shape:          shape = (batch_size,) + input_shape
    elif input_dim:            shape = (batch_size,) + (input_length,) + (input_dim,)
    else:
        # TODO Proper error message
        raise TypeError
    if hidden_dim is None:     hidden_dim = output_dim

    _input = Input(batch_shape=shape)
    _input._keras_history[0].supports_masking = True

    encoder = RecurrentSequential(unroll=unroll, 
                                  stateful=stateful,
                                  return_sequences=True)
    encoder.add(LSTMCell(hidden_dim, batch_input_shape=(shape[0], shape[2])))

    for _ in range(1, depth[0]):
        encoder.add(Dropout(dropout))
        encoder.add(LSTMCell(hidden_dim))

    if bidirectional:
        encoder = Bidirectional(encoder, merge_mode='sum')
        encoder.forward_layer.build(shape)
        encoder.backward_layer.build(shape)
        # patch
        encoder.layer = encoder.forward_layer

    encoded = encoder(_input)
    decoder = RecurrentSequential(decode       = True, 
                                  output_length= output_length,
                                  unroll       = unroll, 
                                  stateful     = stateful)
    decoder.add(Dropout(dropout, batch_input_shape=(shape[0], shape[1], hidden_dim)))
    
    recurrent_decoder_cell = AttentionDecoderCell(
            output_dim = output_dim, 
            hidden_dim = hidden_dim,
            name       = 'attention_map'
            )
    
    if depth[1] == 1:
        decoder.add(recurrent_decoder_cell)
    else:
        decoder.add(recurrent_decoder_cell)
        for _ in range(depth[1] - 2):
            decoder.add(Dropout(dropout))
            decoder.add(LSTMDecoderCell(output_dim=hidden_dim, hidden_dim=hidden_dim))
        decoder.add(Dropout(dropout))
        decoder.add(LSTMDecoderCell(output_dim=output_dim, 
                                    hidden_dim=hidden_dim,
                                    activation = 'softmax'))
    
    inputs         = [_input]
    decoded        = decoder(encoded)
    model          = Model(inputs, decoded)
    
    # layers providing transparency into model operations
    #attention_map  = recurrent_decoder_cell.build_model(recurrent_decoder_cell.)
    #attention_flow = Model(inputs, recurrent_decoder_cell)
    
    return model #, attention_flow

#np.random.seed(12345)
np.random.seed(123456)

model = AttentionSeq2Seq(
        input_dim        = num_distinct_input_tokens, 
        input_length     = max_tokens_per_input_sentence, 
        hidden_dim       = dims[0], 
        output_length    = max_tokens_per_output_sentence, 
        output_dim       = num_distinct_output_tokens, 
#        depth            = (1,1),
        depth            = (1,2),
        bidirectional    = True,
        dropout          = 0
        )

print(model.summary())

#print(attention_flow.summary())
#attention_flow = model.layers.pop()
#attention_flow.summary()
#for layer in model.layers: print (layer.)
#model.get_layer('bidirectional_13')
#model.get_layer(0x2a055828080)

###############################################################################
# Prepare model optimization routine and fit the model
###############################################################################
np.random.seed(1234)

#default_sgd_optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Consider using keras.callbacks.LearningRateScheduler(schedule)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
       if hasattr(self, 'losses') ==False:
           self.losses   = []
           #self.accuracy = []
           #self.lr       = []
 
    def on_epoch_end(self, batch, logs={}):
       self.losses.append(logs.get('loss'))
       #self.accuracy.append(logs.get('accuracy'))
       #self.lr.append(step_decay(len(self.losses)))

global_loss_history = LossHistory()
       
       
#model.load_weights('saved_models/best.hdf5', by_name=False)
def compile_and_fit_the_model(
        initial_epoch    = 0,
        epochs           = 10,
        batch_size       = 64,
        verbose          = True,
        new_optimizer    = None,
        load_weights     = False,
        file_label       = 'best',
        filter_categories= {'CARDINAL'}
        ):
    
    df_train_subset = df_train
    df_val_subset   = df_val
    
    if len(filter_categories)>0:
        # TODO: Subset original data to provided filter, then re-index
        # This may allow the model to learn from simpler (smaller) datasets first
        type_match_new  = lambda type: type in filter_categories
        df_train_subset = df_train[df_train['class'].apply(type_match_new)==True].reset_index(drop=True)
        
        df_val_subset   = df_val[df_val['class'].apply(type_match_new)==True].reset_index(drop=True)

    subset_data_size_train = df_train_subset.shape[0]
    subset_data_size_val   = df_val_subset.shape[0]
    
    print ('Training on %i observations '%subset_data_size_train,
           '\nValidating on %i observations'%subset_data_size_val)
    
    #print(df_train_subset[-25:-1])
    #save_data_frame_to_Excel(df_train_subset[0:65530],'df_train_subset.xls')

    if new_optimizer: #compile optimization if new optimizer is provided
        model.compile(
            loss              = 'categorical_crossentropy', # reward ranking observed translations as likely
            optimizer         = new_optimizer, 
            sample_weight_mode= 'temporal',
            weighted_metrics  = ['accuracy'])    
    
    filename      = 'saved_models/' + file_label + '.hdf5'
    checkpointer1 = ModelCheckpoint(
            save_best_only= True,
            verbose       = 1, 
            filepath      = filename)
    if load_weights==True:
        print('Loading model weights...')
        model.load_weights(filename, by_name=False)    
        print('Done')
   
    model.fit_generator(
            generator           = training_data_generator(batch_size, df_train_subset, False), #input/output arrays
            validation_data     = validation_data_generator(batch_size, df_val_subset),
            epochs              = epochs,
            callbacks           = [checkpointer1, global_loss_history], 
            verbose             = verbose,
            steps_per_epoch     = int(subset_data_size_train/ batch_size),
            shuffle             = False,
            max_queue_size      = 64,
            validation_steps    = int(subset_data_size_val  / batch_size),
            initial_epoch       = initial_epoch
            )
    return 



from keras.optimizers import *
# Default optimizers settings:
# keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
#
# common settings available to all optimizers: clipnorm, clipvalue
# 

# Try Adam Optimizer
#adam_optimizer      = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=10.)
#adam_optimizer_fast = keras.optimizers.Adam(lr=0.5  , beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #, clipnorm=10. #consider clipping norm later
#model.load_weights('saved_models/best.hdf5')
#compile_and_fit_the_model(epochs = 1 , initial_epoch = 0 , optimizer = adam_optimizer_fast, filter_categories= {'CARDINAL'}, batch_size = 32 , compile_needed   = True)
#compile_and_fit_the_model(epochs = 10, initial_epoch = 1 , optimizer = adam_optimizer_fast, filter_categories= {'CARDINAL'}, batch_size = 64                          ) 
#compile_and_fit_the_model(epochs = 12, initial_epoch = 10, optimizer = 'rmsprop'          , filter_categories= {'CARDINAL'}, batch_size = 1                           ) 

# Try RMSProp Optimizer - finds local minimum with below settings
#compile_and_fit_the_model(epochs = 1   , initial_epoch = 0  , batch_size = 16, new_optimizer = RMSprop(lr=0.25)    ) 
#compile_and_fit_the_model(epochs = 5   , initial_epoch = 1  , batch_size = 32) 
#compile_and_fit_the_model(epochs = 50  , initial_epoch = 5  , batch_size = 64) 
#compile_and_fit_the_model(epochs = 100 , initial_epoch = 50 , batch_size = 1024) # doesn't help improve the model
#compile_and_fit_the_model(epochs = 110 , initial_epoch = 100, batch_size = 128 ) 
#compile_and_fit_the_model(epochs = 120 , initial_epoch = 110, batch_size = 128 , new_optimizer = RMSprop(lr=0.05)    ) 
#compile_and_fit_the_model(epochs = 130 , initial_epoch = 120, batch_size = 256 , new_optimizer = RMSprop(lr=0.01, decay = 1e-5)    ) #making gradient calculation more and more precise
#compile_and_fit_the_model(epochs = 200 , initial_epoch = 130, batch_size = 256 , new_optimizer = RMSprop(lr=0.02, decay = 1e-5), verbose = False    ) # hope for faster convergence .. 2.41474 to 2.41400
#compile_and_fit_the_model(epochs = 766 , initial_epoch = 200, batch_size = 64 , new_optimizer = RMSprop(lr=0.05, decay = 1e-5), verbose = False    ) # hope for faster convergence .. 2.41474 to 2.41400

# Try SGD
#compile_and_fit_the_model(epochs = 20   , initial_epoch = 0   , batch_size = 32, new_optimizer = SGD(lr=0.05, momentum=0.8, decay=1e-4, nesterov=True)    )  
#compile_and_fit_the_model(epochs = 20   , initial_epoch = 10  , batch_size = 32)

# Try RMSProp optimizer with default settings, batch size of 1 observation, and go through the training data 10 times
compile_and_fit_the_model(epochs = 10   , initial_epoch = 0  , batch_size = 1, new_optimizer = 'rmsprop') 
compile_and_fit_the_model(epochs = 20   , initial_epoch = 10  , batch_size = 128) 
compile_and_fit_the_model(epochs = 30   , initial_epoch = 20  , batch_size = 256) 
compile_and_fit_the_model(epochs = 50   , initial_epoch = 30  , batch_size = 1024) 
compile_and_fit_the_model(epochs = 100   , initial_epoch = 50  , batch_size = 1024) 
compile_and_fit_the_model(epochs = 500   , initial_epoch = 100  , batch_size = 256, new_optimizer = RMSprop(lr=0.0001)) 
#load best model calibrated yet, as per validation sample


model.load_weights('saved_models/best_Nov5.hdf5')
compile_and_fit_the_model(epochs = 600   , initial_epoch = 500  , batch_size = 256, new_optimizer = RMSprop(lr=0.0001)) 
#compile_and_fit_the_model(epochs = 700   , initial_epoch = 600  , batch_size = 256, new_optimizer = SGD(lr=0.00001)) # trained epochs 600-634
model.load_weights('saved_models/best.hdf5')
compile_and_fit_the_model(epochs = 650   , initial_epoch = 635  , batch_size = 256, new_optimizer = RMSprop(lr=0.0001)) 
model.load_weights('saved_models/best.hdf5')
compile_and_fit_the_model(epochs = 700   , initial_epoch = 650  , batch_size = 256, new_optimizer = RMSprop(lr=0.0001)) 

model.load_weights('saved_models/best_Nov5.hdf5')
compile_and_fit_the_model(epochs = 706   , initial_epoch = 700  , batch_size = 256, new_optimizer = RMSprop(lr=0.0001)) 
compile_and_fit_the_model(epochs = 800   , initial_epoch = 706  , batch_size = 32, new_optimizer = RMSprop(lr=0.0001)) 



###############################################################################
# TODO: visualize model performance over time (weighted accuracy, loss)
###############################################################################
def plot_training_history(
        series      = global_loss_history.losses, 
        x_axis_name = 'epoch',
        y_axis_name = 'categorical cross-entropy loss'
        ):
    plt.plot(range(len(series)), series)
    #plt.legend(loc=2)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)


plot_training_history(global_loss_history.losses[0:10])    
plot_training_history(global_loss_history.losses[0:30])    
    
#model.evaluate_generator(generator, 100)


###############################################################################
# Inference mode (sampling).
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
#    and a "start of sequence" token as target.
#    Output will be the next target token
# 3) Append the target token and repeat
###############################################################################


def sort_and_plot(array_of_values = None, lookup_labels = None, top_N = 5):
    #plot 10 highest value elements in the dictionary
    
    dict1 = dict([(label, value)  for value, label in zip(array_of_values, lookup_labels)])
    #print(dict1)
    dict2 = dict(sorted(dict1.items(), key=itemgetter(1), reverse = True)[0:top_N])
    #print(dict2)
    elements_to_plot = min(top_N, len(dict2))
    ind = np.arange(elements_to_plot,0,-1)
    
    plt.clf()
    plt.barh(ind, dict2.values())
    plt.yticks(ind, dict2.keys())
    plt.show()
    return

#sort_and_plot([0.1, 0.5, 0.4], ['a','b','c'])

def decode_sequence(input_seq = None, graph_likely = True, show_attention_flow = True):
    # Encode the input as state vectors.
    #states_value = encoder_model.predict(input_seq)
    
    # Generate empty target sequence for storing translated phrase
    target_seq = np.zeros((1, 1, num_distinct_output_tokens))
    
    # May want to later populate the first character of target sequence with the start character.
    #target_seq[0, 0, output_token_index['\t']] = 1.

    #print('Shape of model predictions is:', model.predict(input_seq).shape)
    output_tokens_scores = model.predict(input_seq)[0,:,:]
    #attention_flow       = attention_model.predict(input_seq)[0,:,:]
    #print(output_tokens_scores.shape)
    
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    number_of_words_in_translated_sentence = 0 

    #print(output_tokens_scores)

    while not stop_condition:
        # generate probability scores for each of the output tokens
        # based on decoder model applied to encoded state and target sequence
        #output_token_scores = decoder_model.predict([target_seq] + states_value)
        output_token_scores = output_tokens_scores[number_of_words_in_translated_sentence,:]
        
        #print('Probability scores for this token sum up to %i%%'%int(100*output_token_scores.sum()))
        
        #print('Next token scores', output_token_scores) 
        
        # Sample a token that has largest probability to appear next
        #sampled_token_index = np.argmax(output_token_scores[0, -1, :])
        sampled_token_index = np.argmax(output_token_scores)
        if graph_likely==True: sort_and_plot(output_token_scores, output_token_index)
        sampled_word = reverse_output_token_index[sampled_token_index]

        if number_of_words_in_translated_sentence > 0: decoded_sentence += ' '
            
        decoded_sentence += sampled_word        
        number_of_words_in_translated_sentence += 1
        
        # Exit condition: either hit max length or find stop character.
        if (sampled_word == '\n' or
           number_of_words_in_translated_sentence >= max_tokens_per_output_sentence):
            stop_condition = True

        # Add the sampled word to the sequence
        word_vector = np.zeros((1, 1, num_distinct_output_tokens))
        word_vector[0, 0, sampled_token_index] = 1.
        
#        print ('Best guess token: ', sampled_word, 
#               ' with associated score ', np.max(output_token_scores))

        target_seq = np.concatenate([target_seq, word_vector], axis=1)
                
    #print ('target sequence: ',target_seq)

    return decoded_sentence


###############################################################################
# Prepare functionality to translate input sentence
#   1) vectorize sentence using one-hot encoding for each input symbol
#   2) feed vectorized sentence into the model
#   3) run step-by-step translation by picking most likely token at each step
###############################################################################
def vectorize(sentence):
    # initialize decoder input data as 3D NumPy array of zeroes with dimensions:
    # 1) 1
    # 2) maximum sentence length for decoder
    # 3) number of decoder tokens (destination language)
    vectorized_sentence = np.zeros(
        (1, max_tokens_per_input_sentence, num_distinct_input_tokens),
        dtype='bool')
    
    # fill in encoder input  data representing characters via one-hot encoding
    for t, char in enumerate(sentence):
        if t < max_tokens_per_input_sentence:
            vectorized_sentence[0, t, input_token_index[char]] = 1.
       
    #print('Vectorized sentence has %i tokens'%vectorized_sentence.sum())
    return vectorized_sentence

def translate(sentence = None, graph_likely = False, show_attention_flow = False):
    #vectorized_sentence = vectorize('\t' + sentence) 
    vectorized_sentence = vectorize(sentence.lower()) 
    #print('Number of symbols in vectorized sentence is %i: '%vectorized_sentence.sum())
    decoded = decode_sequence(vectorized_sentence, graph_likely, show_attention_flow)[0:-2] #omit end token
    #print (decoded)
    return decoded
    

# Test translation mechanism
translate('911', True)
translate('X')
translate('1003', True)
translate('1')
translate('2')
translate('23')
translate('5')
translate('asdasda ')
translate('100', True)

translate('12', True)
translate('LXXXVIII', True)
translate('-742,791', True)
#translate('10001')
#translate('1000')
#translate('1005')
#translate('10003')
#translate('20003')
#translate('2222')
#translate('111')
#translate('1201')

###############################################################################
# Test whether recommended translations are reasonable
###############################################################################

# Test whether model predictions depend on inputs
def verify_model_generates_constant_predictions(number_of_predictions = 10,
                                                print_sum_squares     = False,
                                                print_decoded_sentence= False):    
    random_inputs = np.random.lognormal(1.0, 
                        2.0,
                        (number_of_predictions,max_tokens_per_input_sentence, num_distinct_input_tokens)
                        )
    random_model_output = model.predict(random_inputs)
    #print(random_model_output.shape)
    for prediction_index in range(1,number_of_predictions,1):
        curr_prediction = random_model_output[prediction_index  ,:,:]
        prev_prediction = random_model_output[prediction_index-1,:,:]
        
        if print_decoded_sentence == True: 
            curr_input_seq  = random_inputs[prediction_index:prediction_index+1,:,:]
            #print (curr_input_seq.shape)
            decode_sequence(curr_input_seq, False)[0:-2]
        
        if print_sum_squares ==True:
            print('Sum of squares of probabilities for prediction number %i is %2f'%(prediction_index, np.multiply(curr_prediction,curr_prediction).sum()))        
        equality_array = np.equal(curr_prediction, prev_prediction)
        #equality_array = np.equal([0, 1, 3], np.arange(3))
        if equality_array.min()==True:
            print('Model probability estimates for prediction #%i coincides with previous one.'%prediction_index)
            
    return
              
verify_model_generates_constant_predictions(10, True)        
verify_model_generates_constant_predictions(10)


#model.load_weights('saved_models/history_Oct29.hdf5')
#model.load_weights('saved_models/history.hdf5')
#model.load_weights('saved_models/recognizing_Kaggle_1_2_sym2word_98pctaccuracy.hdf5')
#model.load_weights('saved_models/history.hdf5')


decode_sequence(np.ones((1,25,67), dtype = bool))


def validate_only(observations = 1, filter_categories = 'CARDINALS'):
    if len(filter_categories)>0:
        type_match_new  = lambda type: type in filter_categories
        df_train_subset = df_train[df_train['class'].apply(type_match_new)==True].reset_index(drop=True)
        token_counter = lambda s: len(s.split()) + 1
        df_train_subset['count_output_tokens'] = df_train_subset['after'].map(token_counter)
        df_train_subset.to_csv('training_subset.csv')
        
    return model.evaluate_generator(
            training_data_generator(1, df_train_subset, False),
                    observations)

#validate_only(100)

###############################################################################
# Review attention flow during translation
###############################################################################


###############################################################################
# Evaluate accuracy of translation on the modeling and testing data
# Define function to evaluate accuracy on the dataset provided
# 1) execute translation
# 2) evaluate accuracy
###############################################################################

model.load_weights('saved_models/best_Nov5.hdf5')

from pandas import ExcelWriter
def save_data_frame_to_Excel(dataset, save_data_to = 'saved.xls'):
    writer = pd.ExcelWriter(save_data_to)
    sheet_name = 'Sheet1'
    dataset.to_excel(writer,sheet_name)
    writer.save()
    return

translation_process = lambda sentence: translate(sentence)

def translate_dataset(dataset = None, save_data_to = 'translated.xlsx', append = False):        
    #print(dataset['before'])
    dataset['translated']       = dataset['before'].apply(translation_process)
    dataset['match']            = dataset['translated'] == dataset['after']
    dataset['max_token_length'] = dataset['before'].apply(count_input_tokens)    
    
    number_of_phrases = dataset.shape[0]
    number_of_phrases_correctly_tranlsated = dataset.loc[dataset['translated'] == dataset['after']].shape[0]
    translation_accuracy = number_of_phrases_correctly_tranlsated / number_of_phrases 
    print('Translation accuracy: %4f'%translation_accuracy)
    
    #dataset.to_csv(save_data_to)
    save_data_frame_to_Excel(dataset, save_data_to)
    return


translate_dataset(df_test, save_data_to = 'test_translated_Nov8v4.xls')
#translate_dataset(dataset = df_val)
#translate_dataset(df_train)

#print(reverse_input_token_index)
