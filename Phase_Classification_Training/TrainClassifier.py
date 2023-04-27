"""
This class trains a classifier network on the dataset in csv format
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import time
import tensorflow as tf
sys.path.append('..\\utils\\')
import common as com
from common import myprint
from tensorflow.keras.layers import Dense, Conv2D, Conv1D, Dropout, BatchNormalization, Flatten, MaxPool2D, MaxPool1D
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# class defined for classifier models
class TrainClassifier():
   def __init__(self, datapath, outpath ='', epochs=300, batch_size=64, dropout_rate = 0.2, batch_norm= True, init = 'glorot_normal', get_time=1,
                loss = CategoricalCrossentropy(), opt='adam', padding = 'same', pool_size = 0, metrics=['accuracy'], strides =(2, 1),
                layers=3, kernel_size=5, filters=3, dense_size =64, activation='relu', out_activation = 'softmax', use_bias =0,        # params for arch
                classes =['p', 's', 'n'], arch='cnn', norm=0, split_ratio=[8, 1, 1], callbacks= ['early_stopping', 'reduce_lr', 'tensorboard']):   #  split_ratio=[train, test, val] 
     self.datapath = datapath 
     self.outpath = outpath         
     self.epochs = epochs         # sampling frequency of the input data
     self.time = get_time
     self.batch_size = batch_size     # input window size in seconds
     self.dropout_rate = dropout_rate      # if input should be filtered
     self.layers=layers    # if single pick for the whole trace
     self.kernel_size = kernel_size
     self.strides = strides
     self.padding = padding
     self.batch_norm= batch_norm
     self.pool_size = pool_size
     self.dense_size = dense_size
     self.use_bias = use_bias
     self.init = init
     self.opt=opt       
     self.callbacks=[]
     self.callback_names=callbacks
     for callback in self.callback_names:
        if callback == 'reduce_lr':
          reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.0001,verbose=1)   
          self.callbacks.append(reduce_lr)
        elif callback == 'early_stopping':
          early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=2, mode='auto')
          self.callbacks.append(early_stopping)
        elif callback == 'tensorboard':               
          log_dir = os.path.join(self.outpath, 'tf_log')
          com.safe_mkdir(log_dir)
          self.tensorboard= tf.keras.callbacks.TensorBoard(log_dir=log_dir)
          self.callbacks.append(self.tensorboard)
     self.classes = classes
     self.norm = norm
     self.metrics = metrics
     self.loss=loss
     self.activation = activation
     self.out_activation = out_activation
     if isinstance(filters, list):
       if len(filters)==self.layers:
         self.filters=filters
     elif isinstance(filters, int):
         self.filters= self.get_filters(filters)
     self.arch = arch
     self.split_ratio = split_ratio if sum(split_ratio)==1 else [r/sum(split_ratio) for r in split_ratio]
     self.input, self.output = self.get_data_csv(self.datapath)
     if self.arch == 'cnn':
       self.model= self.build_CNN_Classifier()
       self.model.summary()
     else:
       self.model= None  
     self.history = None
     self.train_time=0
     self.acc=0
   """
   datapath  : path where dataset is located

   epochs    : Maximum number of training epochs

   batch_size: batch size

   dropout_rate: drop out rate

   batch_norm: if using batchnorm layers

   init : weight initialization

   time : if time is saved

   loss : the loss function to use

   opt  : optimizer

   padding: padding used

   pool_size: size for pooling operation

   metrics : metrics to evaluate

   strides : strides for conv layers

   layers  : number of convolutional layers

   kernel_size: kernal size for convolutional layers

   filters : filters for conv layers

   dense_size: size of the dense layer

   activation: activation for conv layers

   out_activation: activation for dense layer

   use_bias: if bias is used for conv layers

   classes: classes to be predicted 'p', 's', 'tail' and 'n' are currently implemented

   arch: type of network used; 'cnn': conv network; 'ann': dense network

   norm: normalization configuration to use

   split_ratio: [train, val, test] ratios

   callbacks: call backs used in training
   """
   def get_filters(self, filters):
     """
     get a list of filters according to different modes

     1: ascending val*i
  
     2: ascending val*np.power(2, i)

     3: descending val*np.power(2, i)

     returns a list of number of filters used in conv. layers
     """
     val = 64   # base value
     if filters==1: # ascending 
       filter_list =[]  
       for i in range(1, self.layers+1):
         filter_list.append(val*i)
     if filters==2:  # ascending
       filter_list =[]  
       for i in range(0, self.layers):
         filter_list.append(val*np.power(2, i))
     if filters==3: # descending
       filter_list =[]  
       for i in range(self.layers-1, -1, -1):
         filter_list.append(val*np.power(2, i))
     if filters==4: # descending with smaller sizes
       val=16
       filter_list =[]  
       for i in range(self.layers, 0, -1):
         filter_list.append(val*i)
     
     return filter_list

   def class_to_num(self, darray):
    """
    This function converts a list or array of class identifiers to integers

    darray : an array of class id's

    returns an array of integers representing the class
    """
    nums = np.arange(len(self.classes))
    for label, num in zip(self.classes, nums):
       darray[darray==label] = num
    return darray

   def get_data_csv(self, csvpath=None):
     """
     This function reads the data from a CSV format

     csvpath  : path to the csv dataset

     returns two arrays representing inputs and outputs
     """
     
     # Read CSV file into DataFrame df
     if not csvpath: csvpath = self.datapath
     df = pd.read_csv(csvpath+"dataset.csv") # read data
     cols = list(df.columns.values)
     rm_cols = []
     for col in np.arange(len(cols)):
       if 'data_' in cols[col]:              # select columns with data in the label as data columns are labeled as 'data_x_y_z'
         res = [int(i) for i in cols[col].split('_') if i.isdigit()] # get the size of the array dimensions
       else:
         rm_cols.append(cols[col])           # columns not included in data
       
     x = df.drop(list(rm_cols), axis='columns')  # read only data columns as x
  
     if self.norm: x = x.div(x.max(axis=1), axis=0) # if data is to be normalized
     x = np.array(x)  
    
     if len(res)==2: x= np.reshape(x, (-1, res[0]+1, res[1]+1)) # reshape x to correct dimensions
     #if len(res)==1: x= np.reshape(x, (-1, res[0]+1, res[1]+1)) # reshape x to correct dimensions
     
     if 'cnn' in self.arch : x = np.expand_dims(x, axis=-1) # cnn requires a certain shape
     self.input_shape=x.shape[1:] # take the required input shape
     if len(self.input_shape)==2: self.arch = self.arch+'1d'
     y = np.array(df.phase)  # the actual phase of the data is assigned to y
     y = self.class_to_num(y) # phase is converted to corressponding integer
     y = to_categorical(y, num_classes=len(self.classes)) # converted to one hot
     print(f'{x.shape[0]} samples loaded with input shape {x.shape[1:]} and output shape {y.shape[1:]}')     
     return x, y

   def divide_data(self):
     """
     This function divides the input and output arrays according to the ratio provided in the param split_ratio.
     
     Six arrays are returned as train, validation and test arrays for inputs and outputs.
     """
     x_train, x_val, x_test, y_train, y_val, y_test = com.divide_data(self.input, self.output, self.split_ratio)
     
     print(f'training samples are {x_train.shape[0]}, validation samples are {x_val.shape[0]} and test samples are {x_test.shape[0]}')      
     return x_train, x_val, x_test, y_train, y_val, y_test
   
   def fit(self, x_train, y_train, x_val, y_val):
     """
     This function trains the network according to the training data.
     """
     if self.time: t1 = time.time()
     self.history=self.model.fit(x_train, y_train, batch_size = self.batch_size, epochs = self.epochs, use_multiprocessing=True, validation_data=(x_val,y_val), callbacks=self.callbacks)
     if self.time: self.train_time = time.time() - t1
     
   def build_CNN_Classifier(self):
     """
     This function builds a convolutional network according to defined configuration.

     returns a CNN model
     """
     # Initialising 
     CNN_classifier = Sequential()  
     CNN_classifier.add(Conv2D(filters=self.filters[0], kernel_size=self.kernel_size,
                                     padding=self.padding, activation=self.activation, use_bias=self.use_bias, input_shape=self.input_shape))
     #Hidden Layers
     for i in np.arange(1, self.layers):
       CNN_classifier.add(Conv2D(filters=self.filters[i], kernel_size=self.kernel_size, strides=self.strides,
                                    padding=self.padding, activation=self.activation, use_bias=self.use_bias))
       if self.dropout_rate:
         CNN_classifier.add(Dropout(self.dropout_rate))
       if self.batch_norm:
         CNN_classifier.add(BatchNormalization())
     if self.pool_size:
       CNN_classifier.add(MaxPool2D(pool_size=self.pool_size))
     CNN_classifier.add(Flatten())
     CNN_classifier.add(Dense(units=self.dense_size, kernel_initializer = self.init, activation=self.activation))
     if self.dropout_rate:
       CNN_classifier.add(Dropout(self.dropout_rate))  

     #Output Layers (softmax for multi class prediction)
     CNN_classifier.add(Dense(units=len(self.classes), kernel_initializer  = self.init, activation = self.out_activation))
     #Compile ANN
     CNN_classifier.compile(optimizer = self.opt, loss = self.loss, metrics = self.metrics)
     return CNN_classifier

   def build_CNN_Classifier_1d(self):
     """
     This function builds a convolutional network according to defined configuration.

     returns a CNN model
     """
     # Initialising 
     CNN_classifier = Sequential()  
     CNN_classifier.add(Conv1D(filters=self.filters[0], kernel_size=self.kernel_size,
                                     padding=self.padding, activation=self.activation, use_bias=self.use_bias, input_shape=self.input_shape))
     #Hidden Layers
     for i in np.arange(1, self.layers):
       CNN_classifier.add(Conv1D(filters=self.filters[i], kernel_size=self.kernel_size, strides=self.strides,
                                    padding=self.padding, activation=self.activation, use_bias=self.use_bias))
       if self.dropout_rate:
         CNN_classifier.add(Dropout(self.dropout_rate))
       if self.batch_norm:
         CNN_classifier.add(BatchNormalization())
     if self.pool_size:
       CNN_classifier.add(MaxPool1D(pool_size=self.pool_size))
     CNN_classifier.add(Flatten())
     CNN_classifier.add(Dense(units=self.dense_size, kernel_initializer = self.init, activation=self.activation))
     if self.dropout_rate:
       CNN_classifier.add(Dropout(self.dropout_rate))  

     #Output Layers (softmax for multi class prediction)
     CNN_classifier.add(Dense(units=len(self.classes), kernel_initializer  = self.init, activation = self.out_activation))
     #Compile ANN
     CNN_classifier.compile(optimizer = self.opt, loss = self.loss, metrics = self.metrics)
     return CNN_classifier

   def eval(self, x_test, y_test):
    """
    This function evaluates the network usinhg test data
    """
    res= self.model.evaluate(x_test, y_test, batch_size=2* self.batch_size)
    return res

   def plot_acc(self, valpath):
     """
     This function plots the Train-Test Accuracy

     valpath : path to save the plot
     """
     plt.figure()
     plt.plot(self.history.history['accuracy'], label='Training acc')
     plt.plot(self.history.history['val_accuracy'], label='Testing acc')
     plt.xlabel('epochs')
     plt.ylabel('accuracy')
     plt.legend()
     plt.savefig(os.path.join(valpath, 'train_acc.png'))

   def plot_loss(self, valpath):
     """
     This function plots the Train-Test losses

     valpath : path to save the plot
     """
     plt.figure()
     plt.plot(self.history.history['loss'], label='Training loss')
     plt.plot(self.history.history['val_loss'], label='Testing loss')
     plt.xlabel('epochs')
     plt.ylabel('loss')
     plt.legend()
     plt.savefig(os.path.join(valpath, 'train_loss.png'))

   def get_predicted(self, input):
     """
     This function predicts the class for the given input

     input: the given input seismic data

     returns the predicted class
     """
     y_pred=self.model.predict(input)
     return y_pred

   def get_confusion(self, y_act, y_pred):
     """
     This function computes the confusion matrix given the set of true and predicted outputs

     y_act : actual class

     y_pred : predicted class
  
     returns confusion matrix and corressponding dataframe
     """
     labels = np.arange(len(self.classes))
     cm = confusion_matrix(np.argmax(y_act,axis=1), np.argmax(y_pred,axis=1), labels=labels)
     y_actu = pd.Series(np.argmax(y_act,axis=1), name='Actual')
     y_predict = pd.Series(np.argmax(y_pred,axis=1), name='Predicted')
     df_confusion = pd.crosstab(y_actu, y_predict, margins=True)
     return cm, df_confusion

   def get_report(self, cr, df_confusion, valpath):
     """
     This function write the classification report to screen and text file

     cr: classification report

     df_confusion: confusion matrix as dataframe

     valpath : path to save the plot
     """
     lines=[]    
     lines = myprint(lines, 'classification report')
     lines = myprint(lines, cr)
     lines = myprint(lines, 'confusion matrix')
     lines = myprint(lines, f'{df_confusion}')
     txtfile = os.path.join(valpath, 'validation_score.txt')
     with open(txtfile, 'w') as file:
       file.writelines(lines) 

   def plot_confusion(self, cm, valpath):
     """
     This function plots the confusion matrix as heatmap

     cm:  confusion matrix

     valpath : path to save the plot
     """
     plt.figure()
     sns.heatmap(cm, 
            annot=True,
            xticklabels=self.classes, 
            yticklabels=self.classes,
            )
     plt.ylabel('Predicted')
     plt.xlabel('Actual')
     plt.title('Confusion matrix')
     plt.savefig(os.path.join(valpath, 'confusion.png'))

   def write_config(self, valpath, lines=[]):
    """
    This function writes the configuration to a line as well as append to a list.

    lines = list of text

    valpath : path to save the plot
    """
    arch = [] 
    arch.append(com.decorate(f'{self.arch} Architecture'))
    self.model.summary(print_fn=lambda x: arch.append(x + '\n'))
    short_model_summary = "\n".join(arch)
    com.to_file(arch, os.path.join(valpath, 'architecture.txt')) 
    config =  com.get_class_config(self, [com.decorate('Training Configuration')])
    lines+=config 
    com.to_file(lines, os.path.join(valpath, 'config.txt')) 

def main():
   """
   This is an example to using the code to train a classifier involving the following steps:

   - read the data

   - form the directory structure

   - divide the data into training, validation and test sets

   - train the model

   - plot the training progress

   - evaluate the model

   - compute performance matrics

   In order to use the code  following changes should be made

   - point the datapath to the required dataset

   - change resdir as the location to save results

   - change the params of the TrainClassifier object according to the dataset, required architecture and training strategy

   """
   #datapath = "..\\CSV_datasets\\train_dataset\\PESH_train_100hz__1_40filt_4s_HHallchannels_factor2_nperf1_tails_postnorm_max\\"
   datapath = "..\\CSV_datasets\\train\\PESH\\PESH_train_100hz__1_40filt_4s_HHallchannels_factor2_nperf1_tails_postnorm_max\\"
   datapath = "..\\CSV_datasets\\train\\PESH\\PESH_train_100hz_0filt_4s_HHZchannels_factor2_nperf1_tails_postnorm_unity_temporal\\"
   #datapath = "..\\CSV_datasets\\train\\ISB\\ISB_train_100hz_0filt_4s_HHallchannels_factor2_nperf1_tails_postnorm_max\\"
   datapath = "..\\CSV_datasets\\train\\IRIS\\IRIS_train_40hz_0filt_4s_BHallchannels_factor2_nperf1_tails_postnorm_max\\"
   basepath = 'results'
   resdir = 'IRIS_stft_allchannels_5filt_mfilt3'
   outpath = os.path.join(basepath, resdir)
   com.safe_mkdir(outpath)
   modelpath = os.path.join(outpath, 'saved_model')
   com.safe_mkdir(modelpath)
   valpath = os.path.join(outpath, 'validation')
   com.safe_mkdir(valpath)  
   
   training = TrainClassifier(datapath, outpath=outpath, filters=3, strides=1)
   x, y = training.input, training.output
   x_train, x_val, x_test, y_train, y_val, y_test = training.divide_data()
   if training.model==None:
     print(training.arch)
     if '1d' in training.arch:
       training.model= training.build_CNN_Classifier_1d()
       training.model.summary()
       
   history=training.fit(x_train, y_train, x_val, y_val)
   
   training.model.save(os.path.join(modelpath, 'model'))
   print(f'model saved to {modelpath}')
   
   epoch_tot = len(training.history.history['accuracy'])
   if training.time: t_epoch = training.train_time/epoch_tot
   
   #Train-Test Accuracy plot
   training.plot_acc(valpath)
   training.plot_loss(valpath)

   lines = []
   training_acc, val_acc = training.history.history['accuracy'][-1], training.history.history['val_accuracy'][-1]
   if training.time:  lines = myprint(lines, f'Total training time for {epoch_tot} epochs is {training.train_time:.2f} sec with {t_epoch:.2f} sec / epoch.')
  
   evaluate_results=training.eval(x_test, y_test)
   lines = myprint(lines, f'Testing Loss= {evaluate_results[0]:.4f}  Testing Accuracy= {evaluate_results[1]:.4f}')

   y_pred=training.get_predicted(x_test)
   cm, df_confusion=training.get_confusion(y_test, y_pred)
   cr = classification_report(np.argmax(y_test,axis=1), np.argmax(y_pred,axis=1))
   training.get_report(cr, df_confusion, valpath)
   training.plot_confusion(cm, valpath)
   training.write_config(valpath, lines)
   

   

if __name__ == '__main__':
   main() 