"""
This class trains a classifier to classify the seismic phase
"""
import numpy as np
from scipy.signal import find_peaks
import sys
import os
import tensorflow as tf
#from SeedDataset import SeedDataSet
#from Plotting import Plotting
import common as com
#from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from CapsPhase_utils import *
from keras.models import load_model as keras_load_model
from keras.models import save_model as save
from SeedDataset import SeedDataSet
import common as com

# class defined for classifier models
class CapsPhase():
   def __init__(self, mode='caps', f=100, win = 4, filt=[1, 40], # params for preprocessing
                single=0, pthresh=0.9, sthresh=0.8,            # params for implementation
                sta_lta_check=0, sta_lta_p = 1.2, sta_lta_s = 1.1, sta_win=2, # additional checks
                phases =['p', 's', 'n'], full_trace=1,          # phases present in the model
                channels =['HHN', 'HHE', 'HHZ'],                             # channel used
                model_weights = 'models\\CapsPhase\\model\\CapsPhase_CNNX.h5',                           # model parameters
                proc_level = 'batch', eval_level = 'batch'):    # whether preprocessing or validation is performed on samples or trace
     self.f = f         # sampling frequency of the input data
     self.win = win     # input window size in seconds
     self.filt = filt      # if input should be filtered
     self.single=single    # if single pick for the whole trace
     self.phases = phases
     self.nclasses = len(self.phases)
     self.peak_thresh = {}
     self.channels = channels
     self.mode = mode   # the mode can help select between different models
     try:
       self.model = keras_load_model(model_weights, custom_objects={'PrimaryCap': PrimaryCap,
                                                 'CapsuleLayer':CapsuleLayer,
                                                  'Length':Length,
                                                   'margin_loss':margin_loss})
     except:
       self.model = keras_load_model('models\\CapsPhase\\model\\new_model\\', custom_objects={'PrimaryCap': PrimaryCap,
                                                 'CapsuleLayer':CapsuleLayer,
                                                  'Length':Length,
                                                   'margin_loss':margin_loss})
     
     print(f'The model is loaded from {model_weights}.')     
     for phase in self.phases:
       if phase=='p': self.peak_thresh[phase]=pthresh  # threshold for p class
       if phase=='s': self.peak_thresh[phase]=sthresh  # threshold for p class
       elif phase == 'n': self.peak_thresh[phase]=0.0  # threshold for noise
     self.stat_thresh = {}
     for phase in self.phases:
       if phase=='p': self.stat_thresh[phase]=sta_lta_p  # threshold for p class
       if phase=='s': self.stat_thresh[phase]=sta_lta_s  # threshold for p class
       elif phase == 'n': self.stat_thresh[phase]=0.0  # threshold for noise
     self.phase_lims ={}
     for phase in self.phases:
       if phase=='p': self.phase_lims[phase]=[-2, 2]  # threshold for p class
       if phase=='s': self.phase_lims[phase]=[-2, 2]  # threshold for p class
       elif phase == 'n': self.phase_lims[phase]=[0, 0]  # threshold for noise    
     self.full_trace = full_trace
     self.sta_lta_check=sta_lta_check   
     self.sta_win = sta_win
     self.hist_avg = 0
     self.phase_lim = [-self.win/2, self.win/2]
     self.search_win = 1
     self.proc_level = proc_level
     self.eval_level = eval_level 

   """
   mode: the type of model. currently stftcnn1 to stftcnn6 are implemnted

   f: sampling frequency required by the model

   win: window in sec for input to the model

   filt:  denotes the filter to be used; 0: no filter; [x]: lowpass at x Hz; [x, y]: bandpass between x and y Hz

   single  : 0: the peaks from all slices are retained; 1: single peak per trace is retained

   pthresh : the threshold for P phase

   sthresh : the threshold for S phase

   sta_lta_check : if STA\LTA trigger is to be used

   sta_lta_p     : STA\LTA value to turn on the trigger for P phase

   sta_lta_s     : STA\LTA value to turn on the trigger for S phase

   sta_win=10    : window for STA

   phases  : the phases detected by the model (only phase arrivals are detected)

   full_trace: if whole trace to be evaluated

   channels: channels required by the model

   model_path: if using a model other than predefined then provide the path and adjust other paremetrs by hand

   model_params: architectural params

   proc_level   : level for preprocessing; 'trace': performed on the whole trace; 'slice' or 'batch': performed on the current slice

   eval_level   : level for application; 'trace': performed on the whole trace; 'slice' or 'batch': performed on the current slice    
   """

   def preproc(self, stream, to_numpy=1):
     '''
     This function processes a stream by taking STFT

     stream: stream to be processed

     to_numpy: if the stream is to be converted to numpy

     returns the processed data.     
     '''
     st = stream.copy()
     st.detrend('demean')
     info=com.get_info(st)
     if info['f'] != self.f:
       #print('Stream interpolated from frequency {} to {}.'.format(st[0].stats.sampling_rate, self.f))  
       st.interpolate(sampling_rate=self.f)   
     return self.get_batch(st)

   def get_batch(self, st):
     """
     This function converts the stream to numpy array with required format.

     st: stream

     returns batch in the required format
     """
     fo = 5
     fs = 100  #You need to upsample/downsample if the sampling rate is not 100Hz.
     st1 = st.copy()
     st1[0].data = butter_bandpass_filter_zi(st1[0].data, self.filt[0], self.filt[1], fs, order=fo)
     st1[1].data = butter_bandpass_filter_zi(st1[1].data, self.filt[0], self.filt[1], fs, order=fo)
     st1[2].data = butter_bandpass_filter_zi(st1[2].data, self.filt[0], self.filt[1], fs, order=fo)
     batch =[]
     for tr in st1:
       batch.append(tr.data[:-1])
     
     batch = np.array(batch)
     bshape=batch.shape
     batch = batch.reshape(1, bshape[1], bshape[0])
     
     max_batch = np.max(np.abs(batch), axis=(1, 2))
     batch = batch / max_batch     
     return batch

   def post_proc(self, batch_data, prediction, count, tm, stt, f_act, picks_dict, triggers=[]):
     """
     This function processes the network prediction and saves required data in a dictionary

     batch_data: actual data input

     prediction: model prediction
   
     count: slice number

     tm:starting time of current slice

     stt:starting time of trace

     f_act: actual frequency of trace

     picks_dict: dictionary where picks are to be saved

     triggers: The dictionary containing trigger values

     returns the appended picks dictionary.
     """
     
     #pred = np.argmax(prediction, axis=-1)#[0]      # class as number
     #pred =com.most_frequent(pred)
     #pred_phase = self.phases[pred]
     
     if (max(prediction[0][0]) > self.peak_thresh[self.phases[0]]): 
       pred = 0
     elif (max(prediction[0][1]) > self.peak_thresh[self.phases[1]]): 
       pred = 1
     else:              
       pred = 2  # noise if value is less than threshold  
     
     if self.sta_lta_check:
       if count==0: triggers['sta/lta'][self.mode]=[]
       sta, lta, self.hist_avg = com.get_stalta_stats(batch_data[0], self.hist_avg, 
                             self.f, iter_count=count) #get sta\lta stats    
       trigger = lta/self.hist_avg
       triggers['sta/lta'][self.mode].append(trigger)
       if pred_phase =='p' and (trigger < self.stat_thresh[pred_phase]): pred=self.phases.index('n')  # noise if value is less than threshold 
     phase = self.phases[pred]
     utctime =tm - (self.win/2) # current time
     loc = com.get_loc(utctime, stt, f_act) # get position in original trace
     value = prediction[0][pred] # value of the predicted phase
     
     # save the pick in the dict
     #print(picks_dict[self.mode][phase]['utc'])
     picks_dict[self.mode][phase]['utc'].append(utctime)
     picks_dict[self.mode][phase]['value'].append(value)
     picks_dict[self.mode][phase]['pick'].append(loc) # locus wrt actual trace   
     
     # save the values of all outputs
     for p_cnt, phase in enumerate(self.phases):
       picks_dict[self.mode][phase]['out'].append(prediction[0][p_cnt])  
       picks_dict[self.mode][phase]['out_t'].append(utctime)  
       picks_dict[self.mode][phase]['out_loc'].append(loc)
     return picks_dict, triggers
   
   def get_picks(self, batch_data):
     '''
     This function obtains network prediction

     batch_data: data for prediction
     
     returns the raw network output
     '''
               
     prediction = self.model.predict(batch_data)   # model prediction         
     #print(prediction)
     return prediction  

def main():
   """
   Small working example:
   """
   import obspy
   classifier = CapsPhase(mode='caps')
   #classifier.model.save('models\\CapsPhase\\model\\new_model')
   # The path to data
   datapath = '2019_test_traces\\*.mseed'
   channels = classifier.channels  
   # arrival times for the trace
   ptime = 60
   stime= 200
   dataset = SeedDataSet(datapath, ptime=ptime, stime=stime)
   stream_list = dataset.stream_to_list(f=[100], channels=channels)
   print(len(stream_list))
   # read data in stream  
   

   # form a dictionary in required format
   picks_dict={}

   picks_dict[classifier.mode]={}

   for phase in classifier.phases:

     picks_dict[classifier.mode][phase]={}

     picks_dict[classifier.mode][phase]['utc']=[]

     picks_dict[classifier.mode][phase]['value']=[]

     picks_dict[classifier.mode][phase]['pick']=[]
 
     picks_dict[classifier.mode][phase]['out']=[]
 
     picks_dict[classifier.mode][phase]['out_t'] =[]
 
   # loop through streams
   for count, tr in enumerate(stream_list[:1]):
     
     # get information about stream

     info = com.get_info(tr)

     # slice in required size

     pslice = tr.slice(info['stt']+ ptime+2, info['stt']+ ptime+2+classifier.win) # take a slice of required size 2 seconds into the P phase 

     # process the slice  
  
     pslice_proc = classifier.preproc(pslice)
     print(max(pslice_proc), min(pslice_proc))
     # get prediction from the classifier
     exit()
     pred = classifier.get_picks(pslice_proc)

     # convert one hot to phase name

     pred_phase = classifier.phases[np.argmax(pred)]

     print(f'The predicted phase is {pred_phase}')
     
     # load values in picks dictionary

     picks_dict = classifier.post_proc(pslice_proc, pred, count, info['stt']+ ptime+2, info['stt'], info['f'], picks_dict)
 
     print(f'picks_dict = {picks_dict}') 
   
if __name__ == '__main__':
   main() 