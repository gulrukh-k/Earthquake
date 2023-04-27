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

def main():
   """
   This is an example to using the code to train a model involving the following steps:

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
   datapath = "DATA/Data_stats_STEAD_500kmM4ps10_globalnorm_log_20Hz/"
   datapath = "DATA/Peshawar_data_train_log_global_norm_p_halfsec_4sec/" # data path
   basepath = 'results'
   f_factor=5
   resdir = 'STEAD_dataset_smax_mse_f32_k3_1layer_bn1' # result path
   maxpath = os.path.join(basepath, 's_max')
   outpath = os.path.join(maxpath, resdir)
   # call PredTrain object
   training = pred.PredTrain(datapath, outpath=outpath, out='smax', mode='test', f_factor=f_factor)
   
   if training.mode=='transfer':
     trmodelpath = os.path.join(outpath, 'saved_model')
     transpath = os.path.join(outpath, 'transfer_pesh')   
     com.safe_mkdir(transpath)
     outpath = transpath
     training.outpath = transpath

   modelpath = os.path.join(outpath, 'saved_model')
   if training.mode=='train' or  training.mode=='transfer':   com.safe_mkdir(modelpath)
   valpath = os.path.join(outpath, 'validation')
   if training.mode=='train' or  training.mode=='transfer':   com.safe_mkdir(valpath)  
   valtestpath = os.path.join(outpath, 'test_validation_stead')
   if training.mode=='test':   com.safe_mkdir(valtestpath)  
   
   x, y = training.get_data_csv()
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
    
   training.plot_pred(y_test, y_pred, valpath)
   training.scat_pred(y_test, y_pred, valpath)
   training.write_config(valpath, lines)
   

   

if __name__ == '__main__':
   main() 