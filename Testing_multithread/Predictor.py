"""
A class to convert a SeedDataset object into processed data as CSV file
"""
import numpy as np
import obspy
from obspy.core.stream import Stream
import os
import sys
import random
import common as com
from obspy.signal.trigger import pk_baer, plot_trigger
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Model, load_model

class Predictor:
   def __init__(self, mode='m1', outdir = '', channels=['HHZ'], f=100, filt=0,
                win=4, proc='log', norm='global',
                plot=0, num_plots=10):
     """
     mode: network mode
 
     channels: seismic channels to select

     f: sampling rate for the generated data

     filt: denotes the filter to be used; 0: no filter; [x]: lowpass at x Hz; [x, y]: bandpass between x and y Hz

     win: size in seconds for each data sample
   
     proc: processing for data

     norm: the type of normalization

     plot: if plotting predefined number of data samples from (num_plots); 0: no plots; 1: only histograms are plotted; 2: data is plotted for each event

     num_plots: number of plots to make if plot=1

     return_type: type of data returned;'list' implemented
     
     """
     self.mode = mode
     self.channels = channels
     self.norm= norm    
     self.outdir = outdir
     if outdir != '': com.safe_mkdir(outdir)
     if mode =='m1':
         self.model_path = 'PredModel\\m1\\model'
         self.f = 20
         self.filt=0
         self.win = 4
         self.proc='log'
         self.model= load_model(self.model_path)
         self.model.summary()
         print(f'The model {self.model_path} is loaded.')
     if mode =='t1':
         self.model_path = 'PredModel\\t1\\model'
         self.f = 20
         self.filt=0
         self.win = 4
         self.proc='log'
         self.model= load_model(self.model_path)
         self.model.summary()
         print(f'The model {self.model_path} is loaded.')    
         
     if self.proc == '':
       self.fproc = np.array
       self.finv_proc = np.array
     elif self.proc == 'log10':
       self.fproc = np.log10
       def finv_proc(y):
          return (10 ** y)
       self.finv_proc= finv_proc
     elif self.proc == 'log':
       self.fproc = np.log
       def finv_proc(y):
          return np.exp(y)
       self.finv_proc= finv_proc
     self.plot = plot
     self.num_plots = num_plots
     # global params
     self.global_params ={}
     self.global_params['dmax'] = self.fproc(9000000.0)
     self.global_params['dmin'] = self.fproc(1.0)
     self.global_params['smax'] = self.fproc(9000000.0)
     self.global_params['smin'] = self.fproc(400.0)
     self.global_params['psmax'] = 12000
     self.global_params['psmin'] =  1200   
     
   
   def preproc(self, stream, to_numpy=1, squeeze=1):
     '''
     This function procesess a stream by detrending, filtering and selecting the required data from stream

     stream: stream to be processed

     norm: if data is to be normalized

     f: frequency

     filt: if data is to be filtered

     channels: channels to select

     returns an array of processed data     
     '''
     st = stream.copy()    # make a copy of stream for preprocessing     
    
     st.detrend('demean')  # remove mean
     if self.plot: self.plot_data([np.array(st.data)], self.outdir, label=f'detrend')
     info=com.get_info(stream)
     
     if self.filt: # apply filter if required
     
       if len(self.filt)==1:
         st.filter('lowpass', freq=self.filt[0])
       else: 
         st.filter('bandpass', freqmin=self.filt[0], freqmax=self.filt[1])  # optional prefiltering  
     if self.f and (info['f'] != self.f): # if original frequency is different then interpolate to required frequency
       factor = int(np.ceil(info['f']/self.f))
       print(factor)
       if factor < 1:
         print('Stream interpolated from frequency {} to {}.'.format(info['f'], self.f))  
         st.interpolate(sampling_rate=self.f)
     else:
       factor=1
     # get the data as list
     batch_data = []
     if len(self.channels) > 1:
       for count, ch in enumerate(self.channels):
         if ch[-1] in ['*', '?']:
           ch = ch[-1] 
         for tr in st:
           if (ch in tr.stats.channel) and (len(stft_data) < len(self.channels)):
             batch_data.append(tr.data[:-1])
     elif len(self.channels) == 1:
       if hasattr(st, 'stats'): st=[st]
       for tr in st:
         #tr=st
         ch = self.channels[0]
         if ch[-1] in ['*', '?']:
           ch = ch[:-1]  
         if ch in tr.stats.channel:
           batch_data.append(tr.data[:-1])
     else:
       for tr in st:
         batch_data.append(tr.data[:-1])
     if factor > 1:
        new_batch=[]
        for b in batch_data:
           new_batch.append(np.array(b)[::factor])
        batch_data= new_batch
     
     # reshape to required format
     batch_data = np.array(batch_data)
     if self.plot:
      self.plot_data(batch_data, self.outdir, label=f'original')

     batch_data = np.absolute(batch_data)
     batch_data[batch_data<=1]=1
     
     batch_data = np.expand_dims(batch_data, -1)     
     #batch_data = np.where(batch_data==0, 0, self.fproc(np.absolute(batch_data))) # preproc using selected function
     #batch_data = np.absolute(batch_data)
     #batch_data[batch_data<=1]=1
     batch_data = self.fproc(batch_data)
     if self.plot:
        self.plot_data(batch_data, self.outdir, label=f'abs_{self.proc}')
     # normalize p phase data
     if self.norm=='global':
       dmin, dmax = self.global_params['dmin'], self.global_params['dmax']
     else:
       dmin, dmax =0, 0
     batch_data =com.normalize(batch_data, norm='unity', dmin=dmin, dmax=dmax)
     if self.plot:
       self.plot_data(batch_data, self.outdir, label='normed')  
          
     return batch_data

   def predict(self, input):
      output = self.model.predict(input)
      return output

   def postproc(self, output):
      output = np.squeeze(output)
      if self.mode=='m1':
        output = com.inv_norm(output, norm='unity', dmin=self.global_params['smin'], dmax=self.global_params['smax'])
        print(output)
        print(self.global_params['smin'], self.global_params['smax'])
        output = self.finv_proc(output)
        print(output)
      if self.mode=='t1':
        output = com.inv_norm(output, norm='unity', dmin=self.global_params['psmin'], dmax=self.global_params['psmax'])        
      return output

   def plot_data(self, data, plotpath, label=''):
     """
      This function plots the given data

      data: data as array

      plotpath: path where the plot is to be saved

      label: label used for x axis and file name

      saves the plot to the plotpath with name from label
     """
     plt.figure()    
     data = list(data)
     d_max = np.amax(data)
     d_min = np.amin(data)
     lims= com.get_lim(d_min, d_max, 0.1)
     for i in np.arange(len(data)):
       chan = self.channels[i]
       d = data[i]
       plt.plot(d, label = f'{chan} (size={len(d)}) max={d_max:.4f} min={d_min:.4f}')
     plt.xlabel('npts')
     plt.ylabel(label)
     plt.ylim(lims[0], lims[1]) 
     plt.tight_layout() 
     plt.legend()
     plt.savefig(os.path.join(plotpath, f'{label}.png'))
     plt.close()   

   def get_smax(self, sdata, fr, ps_dist):
     sdata.detrend()
     sdata= np.absolute(sdata.data) # take absolute
     smax = np.max(sdata) 
     pos = np.argmax(sdata)
     pos_t = pos/fr
     dis = ps_dist+pos_t 
     return smax, dis

   def plot_pred(self, y_test, y_pred, valpath, label='', num = 200):
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
     plt.ylabel(label,fontsize=18,color='purple')
     plt.savefig(os.path.join(valpath, label+'_.png'))

   def scat_pred(self, y_test, y_pred, valpath, label=''):
     """
     This function makes a scatter plot for the actual and predicted outputs
     
     y_test: actual output

     y_pred: predicted output

     valpath : path to save the plot
     """
     fig = plt.figure( )
     error = np.absolute(y_test-y_pred)
     lim = 1.1 * max(np.max(y_test), np.max(y_pred))
     plt.scatter(y_test, y_pred, label=f'error = {np.mean(error):.4f}$\pm${np.std(error):.4f}')
     plt.plot([0, lim], [0, lim], label=f'truth', color='r', linestyle = 'dotted')     
     plt.legend()
     plt.xlabel("actual")
     plt.ylabel("predicted")
     plt.savefig(os.path.join(valpath, label+'_scatter.png'))
      
def main():
   """
   A working example for using the class

   # Define data path
   dataPath = "..\\SeedData\\PeshawarData2016_2019\\*\\*.mseed"

   # define channels and frequency to select

   channels=['HHZ']

   f=100

   # if dataset for training or test is to be generated

   dataset_type='train'   

   # define SeedDataSet object based on data path and phase arrival times
 
   dataset = SeedDataSet(dataPath, ptime=60, stime=200) 

   # define DataGenerator object for the dataset with required params
     
   gendata = DataGenerator(dataset, dataset_type=dataset_type, channels=channels, plot=1)

   # ouput path
   basepath = f'..\\Prediction_datasets\\{gendata.dataset_type}_dataset'

   datadir= os.path.join(basepath, gendata.get_name('test_Peshwar_stft'))

   # generate dataset in the output path given

   gendata.generate_dataset(datadir)

   # get configuration of data generation

   config = com.get_class_config(gendata, [com.decorate('Dataset Configuration')]) 

   com.to_file(config, os.path.join(datadir, 'config.txt'))
   """
   datapath = 'data\\SeedTestTraces\\*.mseed'
   #channels=['HHE', 'HHN', 'HHZ']
   channels=['HHZ']
   stream = obspy.read(datapath)
   f=20
   ptime=60
   stime = 200
   plot=1
   outdir = 'results\\testing_pred_check2'
   com.safe_mkdir(outdir)
   mpred = Predictor('m1')
   tpred = Predictor('t1')
   act_m=[]
   act_t = []
   pred_m = []
   pred_t = []
   preds=[]
   for count, tr in enumerate(stream[::]):
     print(f'Stream {count}')
     eventdir = os.path.join(outdir, f'events_{count}')
     com.safe_mkdir(eventdir)
     mpred.outdir = eventdir
     tpred.outdir = eventdir
     tpred.plot=0
     if plot: mpred.plot_data([np.array(tr.data)], eventdir, label=f'full')
     info = com.get_info(tr)
     ps_distance = info['endt']-info['stt'] - ptime - stime
     if ps_distance < 10:
       print('P and S arrivals too close')
       continue
     offset = ptime
     pslice = tr.slice(info['stt']+ offset, info['stt']+ offset+mpred.win) # take a slice of required size 2 seconds into the P phase 
     sslice =  tr.slice(info['endt']- stime, info['endt']-stime + 100)
     smax, ps_dis = tpred.get_smax(sslice, info['f'], ps_distance)
     
     act_m.append(smax)
     act_t.append(ps_dis)
     if plot: mpred.plot_data([np.array(pslice.data)], eventdir, label=f'slice')  
     pslice_proc = mpred.preproc(pslice)
     if plot: mpred.plot_data([np.squeeze(pslice_proc)], eventdir, label=f'slice_processed')
     predm = mpred.predict(pslice_proc)
     preds.append(np.squeeze(predm))
     pred_mag = mpred.postproc(predm)
     
     predt = tpred.predict(pslice_proc)
     pred_time = tpred.postproc(predt) /info['f'] 
     
     print(f'The actual smax={smax:.2f} and actual distance ={ps_dis:.2f} sec')
     print(f'The predicted magnitude is {pred_mag:.2f} and distance is {pred_time:.2f} ')
     pred_m.append(pred_mag)
     pred_t.append(pred_time)
   act_m, pred_m = np.array(act_m), np.array(pred_m)
   act_t, pred_t = np.array(act_t), np.array(pred_t)
   act_m_log = com.normalize(np.log(act_m), norm='unity', dmin=mpred.global_params['smin'], dmax=mpred.global_params['smax'])
   mpred.plot_pred(act_m, pred_m, label='smax', valpath = outdir)
   mpred.plot_pred(act_m_log, preds, label='smax_log', valpath = outdir)
   mpred.plot_pred(act_t, pred_t, label='PStime', valpath = outdir)
   mpred.scat_pred(act_m, pred_m, label='smax', valpath = outdir)
   mpred.scat_pred(act_t, pred_t, label='PStime', valpath = outdir)
   mpred.scat_pred(act_m_log, preds, label='smax_log', valpath = outdir)  

if __name__ == '__main__':
   main() 