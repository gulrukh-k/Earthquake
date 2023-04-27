"""
This class uses the PredTrain class for the prediction of time between P and S max
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import time
import tensorflow as tf
import PredTrain as pred
sys.path.append('..\\utils\\')
import common as com
from common import myprint
from tensorflow.keras.layers import Dense, Conv1D, Dropout, BatchNormalization, Flatten, MaxPool1D
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error

def build_pred_1d(training):
     """
     This function builds a convolutional network according to defined configuration.

     returns a CNN model
     """
     # Initialising 
     model = Sequential()  
     if training.conv_layers > 0: 
       model.add(Conv1D(filters=training.filters[0], kernel_size=training.kernel_size,
                                     padding=training.padding, activation=training.activation, use_bias=training.use_bias, input_shape=training.input_shape))
       if training.dropout_rate:
           model.add(Dropout(training.dropout_rate))
       if training.batch_norm:
           model.add(BatchNormalization())
       if training.pool_size:
         model.add(MaxPool1D(pool_size=training.pool_size))

       #Hidden Layers
       for i in np.arange(1, training.conv_layers):
         model.add(Conv1D(filters=training.filters[i], kernel_size=training.kernel_size, strides=training.strides,
                                    padding=training.padding, activation=training.activation, use_bias=training.use_bias))
         if training.dropout_rate:
           model.add(Dropout(training.dropout_rate))
         if training.batch_norm:
           model.add(BatchNormalization())
         if training.pool_size:
           model.add(MaxPool1D(pool_size=training.pool_size))
       model.add(Flatten())
       model.add(Dense(units=training.dense_size, kernel_initializer = training.init, activation=training.activation))
     else:
       model.add(Dense(units=training.dense_size, kernel_initializer = training.init, activation=training.activation, input_shape=training.input_shape))
       #Hidden Layers
       for i in np.arange(1, training.dense_layers):
         model.add(Dense(units=training.dense_size, kernel_initializer = training.init, activation=training.activation))
     
     if training.dropout_rate:
       model.add(Dropout(training.dropout_rate))  
     if training.batch_norm:
           model.add(BatchNormalization())
     #Output Layers (softmax for multi class prediction)
     model.add(Dense(units=1, kernel_initializer  = training.init, activation = training.out_activation))
     #Compile ANN
     model.compile(optimizer = training.opt, loss = training.loss, metrics = training.metrics)
     return model

def main():
   """
   This is an example to using the code to train a network involving the following steps:

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

   - change the params of the PredTrain object according to the dataset, required architecture and training strategy

   """
   datapath = "DATA/Data_stats_STEAD_500kmM4ps10_globalnorm_log_20Hz/"
   datapath = "DATA/Peshawar_data_train_log_global_norm_p_halfsec_4sec/" # path to data
   datapath = "..//Prediction_datasets//train_dataset//Peshawar_psdis_train_100hz_0filt_4s_HHZchannels_globalnorm_logproc//" # path to data
   f_factor = 5 # if data at lower sampling rate is required
   basepath = 'results'
   resdir = 'Peshawar_new_ps_distance_mse_1layer_8filt_3k_bn0_dropout0_8dense_normed' # path for results
   pspath = os.path.join(basepath, 'ps_distance')
   outpath = os.path.join(pspath, resdir)
   
   training = pred.PredTrain(datapath, out = 'PSdistance', outpath=outpath, mode='train', padding = 'same', conv_layers=1, dense_layers=1, # call the PredTrain with correct params
               dropout_rate=0, pool_size = 0, filters=8, dense_size=8, kernel_size=3, f_factor=f_factor, batch_norm=0)
   
   if training.mode=='transfer':
     trmodelpath = os.path.join(outpath, 'saved_model')
     transpath = os.path.join(outpath, 'transfer_pesh_normed')   
     com.safe_mkdir(transpath)
     outpath = transpath
     training.outpath = transpath
   # build the directory structure
   if training.mode=='train':   com.safe_mkdir(outpath)
   modelpath = os.path.join(outpath, 'saved_model')
   if training.mode=='train' or  training.mode=='transfer':   com.safe_mkdir(modelpath)
   valpath = os.path.join(outpath, 'validation')
   if training.mode=='train' or  training.mode=='transfer':   com.safe_mkdir(valpath)  
   valtestpath = os.path.join(outpath, 'test_validation_all')
   if training.mode=='test':   com.safe_mkdir(valtestpath)  
   
   x, y = training.get_data_csv()
   #y = (y-0.2)/0.8
   #y = y/np.max(y)
   training.output = y
   x_train, x_val, x_test, y_train, y_val, y_test = training.divide_data()
   lines = []
   if training.mode=='transfer':
     training.model = load_model(os.path.join(trmodelpath, 'model'))
     training.model.summary()
     print(f'Model loaded from {trmodelpath}')
     for num, layer in enumerate(training.model.layers[:-2]):
       #layer.set_weights = trained_model.layers[num].weights
       layer.trainable = False
       print(f'The {layer.name} is frozen')
     for num, layer in enumerate(training.model.layers[-2:]):
       layer.trainable = True
       print(f'{layer.name} can be trained')
     training.model.summary()
   if training.mode=='train' or training.mode=='transfer':     
     history=training.fit(x_train, y_train, x_val, y_val)    
     training.model.save(os.path.join(modelpath, 'model'))
     print(f'model saved to {modelpath}')   
     epoch_tot = len(training.history.history['mae'])
     if training.time: t_epoch = training.train_time/epoch_tot
   
     #Train-Test mae plot
     training.plot_mae(valpath)
     training.plot_loss(valpath)
      
     training_mae, val_mae = training.history.history['mae'][-1], training.history.history['val_mae'][-1]
     training_loss, val_loss = training.history.history['loss'][-1], training.history.history['val_loss'][-1]
     lines = myprint(lines, f'MAE training ={training_mae:.4f} validation {val_mae:.4f}')
     lines = myprint(lines, f'{training.loss} training ={training_loss:.4f} validation {val_loss:.4f}')
     if training.time:  lines = myprint(lines, f'Total training time for {epoch_tot} epochs is {training.train_time:.2f} sec with {t_epoch:.2f} sec / epoch.')
   elif training.mode=='test':
     training.model = load_model(os.path.join(modelpath, 'model'))
     training.model.summary()
     print(f'Model loaded from {modelpath}')
     valpath = valtestpath
   
   evaluate_results=training.eval(x_test, y_test)
   lines = myprint(lines, f'Testing Loss= {evaluate_results[0]:.4f}  Testing mae= {evaluate_results[1]:.4f}')

   y_pred=np.squeeze(training.get_predicted(x_test))
   if training.time:  lines = myprint(lines, f'Inference time = {training.inference_time:.4f} msec / event.')
   mape = com.MAPE(y_test, y_pred, multioutput="raw_values")   
   mae = np.average(np.abs(y_test-y_pred))
   lines = myprint(lines, f'MAPE Test= {mape:.4f}  Accuracy Test= {100-mape:.4f}  MAE Test= {mae:.4f}')
   r2 = r2_score(y_test, y_pred)
   mape2= mean_absolute_percentage_error(y_test, y_pred)
   
   mae2 = mean_absolute_error(y_test, y_pred)
   lines = myprint(lines, f'sklearn  r2_score Test= {r2:.4f}  MAPE Test= {mape2:.4f}  MAE Test= {mae2:.4f}')
   print(np.max(y_test), np.max(y_pred))  
   training.plot_pred(y_test, y_pred, valpath)
   training.scat_pred(y_test, y_pred, valpath)
   training.write_config(valpath, lines)
   

   

if __name__ == '__main__':
   main() 