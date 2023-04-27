"""
This class trains a regression network on the dataset, predicting the value of the specified continous parameter
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
from tensorflow.keras.layers import Dense, Conv1D, Dropout, BatchNormalization, Flatten, MaxPool1D
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error

# class defined for classifier models
class PredTrain():
   def __init__(self, datapath, mode='train', out='s_max', outpath ='', epochs=300, batch_size=64, dropout_rate = 0.2, batch_norm= True, init = 'glorot_normal', get_time=1,
                loss = 'mse', opt='adam', padding = 'same', pool_size = 0, metrics=['mae'], strides =1, f_factor=0,
                conv_layers=1, dense_layers=1, kernel_size=3, filters=7, dense_size =8, activation='relu', out_activation = 'relu', use_bias =1,        # params for arch
                arch='cnn1d', norm=0, split_ratio=[8, 1, 1], callbacks= ['early_stopping', 'reduce_lr', 'tensorboard', 'checkpoint']):   #  split_ratio=[train, test, val] 
     self.datapath = datapath 
     self.mode = mode
     self.out = out
     self.outpath = outpath         
     self.epochs = epochs         # sampling frequency of the input data
     self.time = get_time
     self.batch_size = batch_size     # input window size in seconds
     self.dropout_rate = dropout_rate      # if input should be filtered
     self.conv_layers=conv_layers    # 
     self.dense_layers=dense_layers
     self.kernel_size = kernel_size
     self.strides = strides
     self.f_factor=f_factor
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
          min_lr = 0.0001
          reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=min_lr,verbose=1)   
          self.callbacks.append(reduce_lr)
        elif callback == 'early_stopping':
          early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=2, mode='auto')
          self.callbacks.append(early_stopping)
        elif callback == 'tensorboard':               
          log_dir = os.path.join(self.outpath, 'tf_log')
          com.safe_mkdir(log_dir)
          self.tensorboard= tf.keras.callbacks.TensorBoard(log_dir=log_dir)
          self.callbacks.append(self.tensorboard)
        elif callback == 'tensorboard':               
          log_dir = os.path.join(self.outpath, 'tf_log')
          com.safe_mkdir(log_dir)
          self.tensorboard= tf.keras.callbacks.TensorBoard(log_dir=log_dir)
          self.callbacks.append(self.tensorboard)
        elif callback == 'checkpoint':
          check_dir = os.path.join(self.outpath, 'check_points')
          com.safe_mkdir(check_dir)
          self.checkpoint = ModelCheckpoint(check_dir, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
          self.callbacks.append(self.checkpoint)
     self.norm = norm
     self.metrics = metrics
     self.loss=loss
     self.activation = activation
     self.out_activation = out_activation
     if isinstance(filters, list):
       if len(filters)==self.conv_layers:
         self.filters=filters
     elif isinstance(filters, int):
         self.filters= self.get_filters(filters)
     self.arch = arch
     self.split_ratio = split_ratio if sum(split_ratio)==1 else [r/sum(split_ratio) for r in split_ratio]
     self.input, self.output = self.get_data_csv(self.datapath)
     if self.mode == 'train':
       if self.arch == 'cnn1d':
         self.model= self.build_pred_1d()
       self.model.summary()
     else:
       self.model= None  
     self.history = None
     self.train_time=0
     self.acc=0
   """
   datapath  : path where dataset is located

   mode: mode in which to run; 'train': train with the dataset; 'test': test using the trained model; 'transfer': train a pretrained model with current data.
 
   out: the selected output:'s_max': maximum amplitude of S wave; 'PSdistance': distance between P and max of S

   outpath: path for the training results and model

   epochs    : Maximum number of training epochs

   batch_size: batch size

   dropout_rate: drop out rate

   batch_norm: if using batchnorm layers

   init : weight initialization

   get_time : if time is saved

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

   out_activation: activation for the output layer

   use_bias: if bias is used for conv layers

   arch: type of network used; 'cnn': conv network; 'ann': dense network; 'cnn1d': conv network for one channel

   norm: normalization configuration to use

   split_ratio: [train, val, test] ratios

   callbacks: call backs used in training
   """
   def get_filters(self, filters):
     """
     get a list of filters according to different modes specified by filters

     1: ascending val*i
  
     2: ascending val*np.power(2, i)

     3: descending val*np.power(2, i)
    
     any other value: same size filters equal to filters

     returns a list of filter size used in conv_layers
     """
     val = 64   # base value
     if filters==1: # ascending 
       filter_list =[]  
       for i in range(1, self.conv_layers+1):
         filter_list.append(val*i)
     elif filters==2:  # ascending
       filter_list =[]  
       for i in range(0, self.conv_layers):
         filter_list.append(val*np.power(2, i))
     elif filters==3: # descending
       filter_list =[]  
       for i in range(self.conv_layers-1, -1, -1):
         filter_list.append(val*np.power(2, i))
     elif filters==4: # descending with smaller sizes
       val=16
       filter_list =[]  
       for i in range(self.conv_layers, 0, -1):
         filter_list.append(val*i)
     else:
       filter_list = [filters]*self.conv_layers
       
     return filter_list

 
   def get_data_csv(self, csvpath=None):
     """
     This function reads the data from a CSV format

     csvpath  : path to the csv dataset

     returns two arrays representing inputs and outputs
     """
     
     # Read CSV file into DataFrame df
     if not csvpath: csvpath = self.datapath
     try:
       df = pd.read_csv(csvpath+"csv_data.csv") # read data
     except:
       df = pd.read_csv(csvpath+"dataset.csv") # read data
     cols = list(df.columns.values)
     rm_cols = []
     for col in np.arange(len(cols)):
       if ('Z' in cols[col]) or ('data' in cols[col]):              # select columns with data in the label as data columns are labeled as 'data_x_y_z'
         res = [int(i) for i in cols[col].split('_') if i.isdigit()] # get the size of the array dimensions
       else:
         res=0
         rm_cols.append(cols[col])           # columns not included in data
       
     x = df.drop(list(rm_cols), axis='columns')  # read only data columns as x
  
     if self.norm: x = x.div(x.max(axis=1), axis=0) # if data is to be normalized
     if self.f_factor:
       res=0
       x2=[]     
       for i in range(x.shape[0]):
         ele = np.array(x.iloc[i])
         x2.append(ele[::5])
       x = np.array(x2)  
     else:
       x = np.array(x)
     
     if res: x= np.reshape(x, (-1, res[0]+1, res[1]+1)) # reshape x to correct dimensions
     if ('cnn' in self.arch) and (self.conv_layers>0) : x = np.expand_dims(x, axis=-1) # cnn requires a certain shape
     self.input_shape=x.shape[1:] # take the required input shape
     y = np.array(df[self.out])  # the actual phase of the data is assigned to y
     ymin = y.min()
     if ymin<0:
       y = (y - y.min())/(y.max()-y.min())
       print(f'the output minimum changed from {ymin} to {y.min()}')
     
     print(f'{x.shape[0]} samples loaded with input shape {x.shape} and output shape {y.shape}')  
      
     return x, y

   def divide_data(self):
     """
     This function divides the input and output arrays according to the ratio provided in the self.split_ratio.
     
     Six arrays are returned as train, validation and test arrays for inputs and outputs.
     """
     x_train, x_val, x_test, y_train, y_val, y_test = com.divide_data(self.input, self.output, self.split_ratio)
     
     print(f'training samples are {x_train.shape[0]}, validation samples are {x_val.shape[0]} and test samples are {x_test.shape[0]}')     
     self.train_num = x_train.shape[0]
     self.val_num = x_val.shape[0] 
     self.test_num = x_test.shape[0]
     return x_train, x_val, x_test, y_train, y_val, y_test
   
   def fit(self, x_train, y_train, x_val, y_val):
     """
     This function trains the network according to the training data.

     x_train: train input
 
     y_train: train output 

     x_val: validation input

     y_val: validatio output

     The self.model is trained and self.history stores the training results
     """
     if self.time: t1 = time.time()
     self.history=self.model.fit(x_train, y_train, batch_size = self.batch_size, epochs = self.epochs, use_multiprocessing=True, validation_data=(x_val,y_val), callbacks=self.callbacks)
     if self.time: self.train_time = time.time() - t1
     
   def build_pred_1d(self):
     """
     This function builds a convolutional network according to defined configuration.

     returns a CNN model
     """
     # Initialising 
     model = Sequential()  
     if self.conv_layers > 0: 
       model.add(Conv1D(filters=self.filters[0], kernel_size=self.kernel_size,
                                     padding=self.padding, activation=self.activation, use_bias=self.use_bias, input_shape=self.input_shape))
       #Hidden Layers
       for i in np.arange(1, self.conv_layers):
         model.add(Conv1D(filters=self.filters[i], kernel_size=self.kernel_size, strides=self.strides,
                                    padding=self.padding, activation=self.activation, use_bias=self.use_bias))
         if self.dropout_rate:
           model.add(Dropout(self.dropout_rate))
         if self.batch_norm:
           model.add(BatchNormalization())
       if self.pool_size:
         model.add(MaxPool1D(pool_size=self.pool_size))
       model.add(Flatten())
       model.add(Dense(units=self.dense_size, kernel_initializer = self.init, activation=self.activation))
     else:
       model.add(Dense(units=self.dense_size, kernel_initializer = self.init, activation=self.activation, input_shape=self.input_shape))
       #Hidden Layers
       for i in np.arange(1, self.dense_layers):
         model.add(Dense(units=self.dense_size, kernel_initializer = self.init, activation=self.activation))
     
     if self.dropout_rate:
       model.add(Dropout(self.dropout_rate))  

     #Output Layers (softmax for multi class prediction)
     model.add(Dense(units=1, kernel_initializer  = self.init, activation = self.out_activation))
     #Compile ANN
     model.compile(optimizer = self.opt, loss = self.loss, metrics = self.metrics)
     return model

   def eval(self, x_test, y_test):
    """
    This function evaluates the network usinhg test data
    """
    res= self.model.evaluate(x_test, y_test, batch_size=2* self.batch_size)
    return res

   def plot_mae(self, valpath):
     """
     This function plots the Train-Test mae

     valpath : path to save the plot
     """
     plt.figure()
     plt.plot(self.history.history['mae'], label='Training mae')
     plt.plot(self.history.history['val_mae'], label='Testing mae')
     plt.xlabel('epochs')
     plt.ylabel('mae')
     plt.legend()
     plt.savefig(os.path.join(valpath, 'train_mae.png'))

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
     if self.time: t_init = time.time()
     y_pred=self.model.predict(input)
     if self.time: 
       self.inference_time = 1000 * ((time.time() - t_init)/input.shape[0])
     return y_pred

   def plot_pred(self, y_test, y_pred, valpath, num = 200):
     """
     This function plots the actual and predicted outputs

     y_test: actual output

     y_pred: predicted output

     valpath : path to save the plot

     num: maximum number of values to plot
     """
     fig = plt.figure(figsize=(20,5))
     plt.plot(y_test[0:num],color='red', alpha=0.8, linewidth=1, label='True Values')
     plt.plot(y_pred[0:num],color='blue',alpha=0.8,linewidth=1,label='Predicted values')
     plt.legend(loc='upper left',prop={'size':15})
     plt.xlabel("Samples",fontsize=18,color='purple')
     plt.ylabel("S max",fontsize=18,color='purple')
     plt.savefig(os.path.join(valpath, 'output.png'))

   def scat_pred(self, y_test, y_pred, valpath):
     """
     This function makes a scatter plot for the actual and predicted outputs
     
     y_test: actual output

     y_pred: predicted output

     valpath : path to save the plot
     """
     fig = plt.figure( )
     error = np.absolute(y_test-y_pred)
     plt.scatter(y_test, y_pred, label=f'error = {np.mean(error):.4f}$\pm${np.std(error):.4f}')
     plt.plot([0, 1], [0, 1], label=f'truth', color='r', linestyle = 'dotted')     
     plt.legend()
     plt.xlabel("actual")
     plt.ylabel("predicted")
     plt.savefig(os.path.join(valpath, 'scatter.png'))

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
   pass   

if __name__ == '__main__':
   main() 