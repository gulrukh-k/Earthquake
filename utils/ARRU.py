"""
This class implements the ARRU model available at https://github.com/tso1257771/Attention-Recurrent-Residual-U-Net-for-earthquake-detection.
"""

import numpy as np
from scipy.signal import find_peaks
import sys
import os
from models.arru.model.build_model_pick import unets as picker_unets
from models.arru.model.build_model_detect import unets as detector_unets
from SeedDataset import SeedDataSet
import common as com


class AppARRU():
   def __init__(self, mode='arrupick', f=100, win = 20, filt=0, 
                single=0, pthresh=0.7, sthresh=0.7,
                sta_lta_check=0, sta_lta_p = 1.5, sta_lta_s = 1.5, sta_win=10,
                phases =['p', 's'], full_trace=1, 
                picker_weights = 'models\\arru\\weights\\train_pick.hdf5',
                detector_weights = 'models\\arru\\weights\\train_multi.hdf5',
                proc_level = 'trace', eval_level = 'slice'):     
     """ 
     mode    : The mode selects between different versions of the model

     f	     : frequency required by the model 

     win     : window used by the model in seconds  

     filt    : denotes the filter to be used; 0: no filter; [x]: lowpass at x Hz; [x, y]: bandpass between x and y Hz

     single  : 0: the peaks from all slices are retained; 1: single peak per trace is retained

     pthresh : the threshold for P phase

     sthresh : the threshold for S phase

     sta_lta_check : if STA\LTA trigger is to be used

     sta_lta_p     : STA\LTA value to turn on the trigger for P phase

     sta_lta_s     : STA\LTA value to turn on the trigger for S phase

     sta_win=10    : window for STA

     phases  : the phases detected by the model (only phase arrivals are detected)

     picker_weights : model weights for picker model

     detector_weights : model weights for detector model

     proc_level   : level for preprocessing; 'trace': performed on the whole trace; 'slice': performed on the current slice

     eval_level   : level for application; 'trace': performed on the whole trace; 'slice': performed on the current slice
     """ 
     self.f = f 
     self.win = win
     self.preproc = self.arru_proc
     self.picker = picker_unets().build_attR2unet
     self.detector = detector_unets().build_attR2unet
     self.mode = mode 
     if mode =='arrupick':
       self.model= self.picker(picker_weights)
     elif mode == 'arrudetect':
       self.model= self.detector(detector_weights)
     else:
       print(f'The mode {mode} is not available.')
       exit()
     self.filt = filt
     self.single=single
     self.pthresh=pthresh
     self.sthresh=sthresh 
     self.phases = phases
     self.nclasses = len(self.phases)
     self.peak_thresh = {}
     for phase in self.phases:
       if phase=='p': self.peak_thresh[phase]=pthresh  # threshold for p class
       if phase=='s': self.peak_thresh[phase]=sthresh  # threshold for p class
     self.peak_thresh['n']=0.0 # threshold for noise
     self.stat_thresh = {}
     for phase in self.phases:
       if phase=='p': self.stat_thresh[phase]=sta_lta_p  # threshold for p class
       if phase=='s': self.stat_thresh[phase]=sta_lta_s  # threshold for p class
       elif phase == 'n': self.peak_thresh[phase]=0.0  # threshold for noise
     self.full_trace = full_trace
     self.sta_lta_check=sta_lta_check   
     self.sta_lta_p = sta_lta_p
     self.sta_lta_s = sta_lta_s
     self.sta_win= sta_win
     self.hist_avg= 0
     self.search_win = 1
     self.proc_level = proc_level
     self.eval_level = eval_level 

   # stream preprocessed for ARRU
   def arru_proc(self, stream, to_numpy=0):
     """
     This function procesess a stream for ARRU network

     stream: stream to be processed  

     to_numpy: if to convert from stream to array

     return processed data   
     """
     st = stream.copy()
     info=com.get_info(st)
     if info['f'] != self.f:
       #print('Stream interpolated from frequency {} to {}.'.format(st[0].stats.sampling_rate, self.f))  
       st.interpolate(sampling_rate=self.f)   
     if self.filt: st.filter('bandpass', freqmin=filt[0], freqmax=filt[1])  # optional prefiltering  
     st= self.stream_standardize(st)
     # Z-score standardize before making predictions
     if to_numpy:  
        batch_data = self.get_numpy(st)
        return batch_data
     else: 
        return st

   def get_batch(self, st):
     """
     This function converts the stream to numpy array with required format.

     st: stream

     returns batch in the required format
     """
     return np.array([i.data for i in st]).T[np.newaxis, ...]

   def get_model(self, model_dict, weights = 'models\\arru\\weights\\train_pick.hdf5'):
     """
     This function appends a dict with model and associated params

     model_dict  : dict to append the data to

     weights     : weights of the trained model

     returns the updat
     """
     model_dict[self.mode] = {}
     model_dict[self.mode]['model'] = self.model(weights) # model
     model_dict[self.mode]['f'] = self.f #required frequency in Hz for ARRU
     model_dict[self.mode]['window'] = self.win #sec
     model_dict[self.mode]['dt'] = 1.0/self.f
     model_dict[self.mode]['preproc'] = self.preproc
     model_dict[self.mode]['app'] = self
     print(f'ARRU model ({self.mode}) is loaded.')
     return model_dict

   def get_picks(self, batch_data):
     """
     This function predicts phase for a batch of data 

     batch_data: input data

     returns predicted output
     """
     predictions = self.model.predict(batch_data) 
     return predictions

   def post_proc(self, batch_data, prediction, count, tm, stt, f_act, picks_dict, triggers):
     """
     This function pereforms post-processing on the model predictions:

     batch_data: input data
     
     prediction : model prediction

     count : current slice number within the trace

     tm:  strating time of current slice

     stt: starting time of the actual trace

     f_act: frequency of actual stream

     picks_dict: dict to hold the model picks

     triggers: addional triggers to incorporate. Currently only STA/LTA is implemented.

     returns the processed batch
     """  
     if self.mode == 'arrupick':  
       predict = prediction[0].T
       predict = [predict[0], predict[1]]
         
     elif self.mode == 'arrudetect':
       pred_pick = prediction[0]
       pred_mask = prediction[1]
       pred_pick_T = pred_pick[0].T
       predict = [pred_pick_T[0], pred_pick_T[1]]
       predict_nz = pred_pick_T[2]

       pred_mask_T = pred_mask[0].T
       pred_eq_mask = pred_mask_T[0]
       pred_nz_mask = pred_mask_T[1]

     if self.sta_lta_check:
       if count==0: triggers['sta/lta'][self.mode]=[]
       sta, lta, self.hist_avg = com.get_stalta_stats(batch_data[0], self.hist_avg, 
                             self.f, iter_count=count) #get sta\lta stats    
       trigger = lta/self.hist_avg 
       triggers['sta/lta'][self.mode].append(trigger)         
     else:
       trigger = 10     
     
     # Loop over phases
     for p_cnt, phase in enumerate(self.phases):
       # get locus (sec) of true picks wrt current slice       
       if self.full_trace:
         labeled = int((self.win)/2) # window center as center of searh space
         search_win= (self.win/2)    # half of window as width of search space
       else:
         labeled = int(arrivals[p_cnt] - tm)  # center at actual pick
         search_win = self.search_win
       
       if (labeled > 0) and (trigger > self.stat_thresh[phase]):#(np.max(cft) > 1.6):  
         peak, value = self.pick_peaks(predict[p_cnt], labeled, 1/self.f, 
                         search_win=search_win, peak_value_min=self.peak_thresh[phase])
                    
         if peak >0:
           utctime = tm + peak
           loc = com.get_loc(utctime, stt, f_act)
           picks_dict[self.mode][phase]['utc'].append(utctime)
           picks_dict[self.mode][phase]['value'].append(value)
           picks_dict[self.mode][phase]['pick'].append(loc) # locus wrt actual trace
           picks_dict[self.mode][phase]['out'].append(value)
           picks_dict[self.mode][phase]['out_t'].append(loc)
         else:
           picks_dict[self.mode][phase]['out'].append(np.max(predict[p_cnt]))
           loc = com.get_loc(tm, stt, f_act)
           picks_dict[self.mode][phase]['out_loc'].append(loc+ np.argmax(predict[p_cnt]))
           picks_dict[self.mode][phase]['out_t'].append(tm)
       else:
         picks_dict[self.mode][phase]['out'].append(np.max(predict[p_cnt]))
         loc = com.get_loc(tm, stt, f_act)
         picks_dict[self.mode][phase]['out_loc'].append(loc+ np.argmax(predict[p_cnt]))
         picks_dict[self.mode][phase]['out_t'].append(tm)
     return picks_dict, triggers

   def pick_peaks(self, prediction, labeled_phase, sac_dt=None,
                     search_win=1, peak_value_min=0.01):
     """
     This function searches for potential picks    
     
     prediction: predicted functions

     labeled_phase: the timing of actual phase arrival

     sac_dt: delta 

     search_win: time window (sec) for searching local maximum near labeled phases

     peak_value_min: minimum value for a peak to be detected

     returns time and value of peaks with respect to current slice 
     """     
     tphase = int(round(labeled_phase/sac_dt))
     
     search_range = [tphase-int(search_win/sac_dt), 
                        tphase+int(search_win/sac_dt)]
     
     peaks, values = find_peaks(np.squeeze(prediction), height=peak_value_min)
     #print(np.max(prediction), peaks, values)
     if len(peaks >0):
       #print('peaks, values', peaks, values)
       in_search = [np.logical_and(v>search_range[0], 
                        v<search_range[1]) for v in peaks]
       _peaks = peaks [in_search]
        
       _values = values ['peak_heights'][in_search]
       #print('_peaks, _values', _peaks, _values)
       if len(_values)>0:
         return _peaks[np.argmax(_values)]*sac_dt, \
                _values[np.argmax(_values)]
       else:
        return -999, -999
     else:
        return -999, -999

   def stream_standardize(self, st):
     """
     This function processes the stream

     input: obspy.stream object (raw data)

     output: obspy.stream object (standardized)
 
     returns processed stream
     """
     st.detrend('demean')
     for s in st:
       s.data /= np.std(s.data)
     return st

def main():
   """
   Small working example:
   """
   dataPath = "..\\SeedData\\Jan2018-Jan2020_seed\\mseed4testing\\*.mseed"
   stations= ['AKA', 'NIL', 'SIMI']
   channels = ['BH1', 'BH2', 'BHZ']
   f = 40
   dataset = SeedDataSet(dataPath)
   stream_list = dataset.stream_to_list(f=f, channels=channels, stations=stations)
   info = dataset.info
   picker_weights = 'models\\arru\\weights\\train_pick.hdf5'
   detector_weights = 'models\\arru\\weights\\train_multi.hdf5'
   pick_mode = ['arrupick', 'arrudetect']
   model_dict={}
   picks_dict={}
   for mode in pick_mode:
      if 'arru' in mode:
        if mode=='arrupick':
          pthresh, sthresh=0.8, 0.7
        elif mode =='arrudetect':
          pthresh, sthresh=0.5, 0.4
        arru_model = AppARRU(mode, single=0, pthresh=pthresh, sthresh = sthresh)
        model_dict[mode]={}
        model_dict[mode]['model']=arru_model

   lines=[]
   for n, st in enumerate(stream_list[:1]):
     lines = com.myprint(lines, '****************************************************************************')
     lines = com.myprint(lines,'Stream : {} '.format(n))  
     for mode in model_dict:
       picks_dict[mode] = {}  
       picks_dict[mode] = model_dict[mode]['model'].get_picks(st, model_dict, picks_dict, dataset)
   print(picks_dict)
     
if __name__ == '__main__':
   main() 