"""
This class trains a classifier to classify the seismic phase
"""
import numpy as np
from scipy.signal import find_peaks
import sys
import os
import common as com
from tensorflow.keras.models import Model, load_model

# class defined for classifier models
class Classifier():
   def __init__(self, mode='stftcnn', f=40, win = 4, filt=[5], # params for preprocessing
                single=0, pthresh=0.95, sthresh=0.85,            # params for implementation
                sta_lta_check=0, sta_lta_p = 1.2, sta_lta_s = 1.1, sta_win=2, # additional checks
                phases =['n', 'p', 's'], full_trace=1,          # phases present in the model
                channels =['Z'],                              # channel used
                model_path = None,                              # path for saved model or weights
                model_params = {},                              # model parameters
                proc_level = 'batch', eval_level = 'batch'):    # whether preprocessing or validation is performed on samples or trace
     self.f = f         # sampling frequency of the input data
     self.win = win     # input window size in seconds
     self.filt = filt      # if input should be filtered
     self.single=single    # if single pick for the whole trace
     self.phases = phases
     self.nclasses = len(self.phases)
     self.peak_thresh = {}
     self.half= 1
     self.mode = mode   # the mode can help select between different models
     # load model
     if 'stft' in self.mode:        
       self.preproc = self.stft_proc
       if model_path:
         self.model_path = model_path
       elif mode[-1]=='1':
         self.model_path = 'models\\stftcnn1\\saved_model\\model'
         self.f = f
       elif mode[-1]=='2':
         self.model_path = 'models\\stftcnn2\\saved_model\\model.h5'
       elif mode[-1]=='5':
         self.model_path = 'models\\stftcnn5\\saved_model\\model'
         self.f = 100
         self.filt=0
         self.pthresh=0.8
         self.sthresh=0.8
         self.phases =['p', 's', 'n']
         self.half=1
       elif mode[-1]=='6':
         self.model_path = 'models\\stftcnn6\\saved_model\\model'
         self.f = 100
         self.filt=[5]
         self.pthresh=0.8
         self.sthresh=0.8
         self.phases =['p', 's', 'n']
         self.half=1
       elif mode[-1]=='7':
         self.model_path = 'models\\stftcnn7\\saved_model\\model'
         self.f = 100
         self.filt=0
         self.pthresh=0.8
         self.sthresh=0.8
         self.phases =['p', 's', 'n']
         self.half=1
       elif mode[-1]=='8':
         self.model_path = 'models\\stftcnn8\\saved_model\\model'
         self.f = 100
         self.filt=0
         pthresh=0.5
         sthresh=0.5
         self.phases =['p', 's', 'n']
         self.half=1
       else:
         self.model_path = '..\\Phase_Classification_ANN\\results\\fft_filt_5hz_half_stft_4s_tails_3class_3_layer_same_adam_rd_lr_ver2_filt3_dsize64\\saved_model\\model.h5'
       #try:
       self.model= load_model(self.model_path)
       self.model.summary()
       print(f'The model {self.model_path} is loaded.')
         
       if 0:  
         print(f'The model {self.model_path} cannot be loaded.') 
         exit()    
         if 'cnn' in mode:
           from models.classification.model.models_2d import build_CNN_Classifier as build_model
           self.model_params = self.get_cnn_params()
           type='cnn'
         elif 'attcnn' in mode:
           from models.classification.model.models_2d import build_ATTCNN_Classifier as build_model
           self.model_params = self.get_attcnn_params()
           type='attcnn'
         self.model = build_model(input_shape=self.model_params['input_shape'], 
                        layers= self.model_params['layers'], padding=self.model_params['padding'], 
                        kernel_size= self.model_params['kernel_size'], dense_size= self.model_params['dense_size'], 
                        filters=self.model_params['filters'], pool_size=self.model_params['pool_size'])                    
         self.model.load_weights(self.model_path )
         print(f'The weights {self.model_path} are loaded by {type} model.')     
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
       if phase=='p': self.phase_lims[phase]=[-2, 5]  # threshold for p class
       if phase=='s': self.phase_lims[phase]=[-2, 5]  # threshold for p class
       elif phase == 'n': self.phase_lims[phase]=[0, 0]  # threshold for noise    
     
     self.channels = channels
     self.full_trace = full_trace
     self.sta_lta_check=sta_lta_check   
     self.sta_win = sta_win
     self.hist_avg = 0
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

   model_path: if using a model other than predefined then provide the path

   model_params: architectural params

   proc_level   : level for preprocessing; 'trace': performed on the whole trace; 'slice' or 'batch': performed on the current slice

   eval_level   : level for application; 'trace': performed on the whole trace; 'slice' or 'batch': performed on the current slice    
   """
   # A similar function can be created for required params
   def get_cnn_params(self, n=0):
     '''
     get model parameters
     n: model ID     
     '''
     params = {}
     if n==0:
       params['input_shape']=(11, 9, 1)
       params['padding']='same'
       params['kernel_size']=5
       params['dense_size'] = 64
       params['filters'] = [256, 128, 64]
       params['pool_size']=0
       params['layers']=3
     return params

   def get_attcnn_params(self, n=0):
     '''
     get model parameters
     n: model ID     
     '''
     params = {}
     if n==0:
       params['input_shape']=(11, 9, 1)
       params['padding']='same'
       params['kernel_size']=5
       params['dense_size'] = 64
       params['filters'] = [256, 128, 64]
       params['pool_size']=0
       params['layers']=3
     return params

   # stream preprocessed for STFT based approach
   def stft_proc1(self, stream, to_numpy=0):
     '''
     process a stream by taking STFT
     stream: stream to be processed
     
     '''
     st = stream.copy()    # make a copy of stream for preprocessing
     st.detrend('demean')  # remove mean
     if st[0].stats.sampling_rate != self.f: # if original frequency is different then interpolate to required frequency
       print('Stream interpolated from frequency {} to {}.'.format(st[0].stats.sampling_rate, self.f))  
       st.interpolate(sampling_rate=self.f)  
     if self.filt: # apply filter if required
       print('filter')
       if len(self.filt)==1:
         st.filter('lowpass', freq=self.filt[0])
       else: 
         st.filter('bandpass', freqmin=self.filt[0], freqmax=self.filt[1])  # optional prefiltering  

     # get the transform as list
     stft_data = []
     for count, ch in enumerate(self.channels): 
       for tr in st:
         if ch in tr.stats.channel:
           stft_data.append(com.get_stft(tr.data[:-1], self.f, norm=1, half=self.half))
     
     # reshape to required format
     stft_data = com.get_numpy(stft_data)
     stft_data = np.moveaxis(stft_data, 1, -1)
     #print(stft_data.shape )
     #exit()
     return stft_data

   def stft_proc(self, stream, to_numpy=1):
     stft_data = com.stft_proc(stream, norm=1, f=self.f, filt=self.filt, channels=self.channels, half=self.half, to_numpy=to_numpy, squeeze=0)     
     return stft_data

   def post_proc(self, batch_data, prediction, count, tm, stt, f_act, picks_dict, triggers=[]):
     
     pred = np.argmax(prediction, axis=-1)[0]      # class as number
     pred_phase = self.phases[pred]
    
     if (prediction[0][pred] < self.peak_thresh[pred_phase]):        
       pred=self.phases.index('n')  # noise if value is less than threshold  
     
     if self.sta_lta_check:
       if count==0: triggers['sta/lta'][self.mode]=[]
       sta, lta, self.hist_avg = com.get_stalta_stats(batch_data[0], self.hist_avg, 
                             self.f, iter_count=count) #get sta\lta stats    
       trigger = lta/self.hist_avg
       triggers['sta/lta'][self.mode].append(trigger)
       if (trigger < self.stat_thresh[pred_phase]): pred=0
     phase = self.phases[pred]
     utctime =tm + (self.win/2) # current time
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
       picks_dict[self.mode][phase]['out_t'].append(loc)  
     return picks_dict, triggers, phase


   def get_picks(self, batch_data):
     '''
     get picks
     batch_data: data for prediction
     tm: starting time for current batch
     picks_dict:  dict to save picks
     stt: trace starting time
     phases: phases required by the test bench
     f_act: frequency of actual trace
     sta: station
     lta: previous long time average
     hist_avg: previous historical average
     ptime: actual p arrival
     stime: actual s arrival
     '''
               
     prediction = self.model.predict(batch_data)   # model prediction         
     
     return prediction  

def main():
   import obspy
   mode = 'stftcnn5'
   classifier = Classifier(mode= mode)
   
   datapath = 'data\\test_traces\\*.mseed'
   stream = obspy.read(datapath)
   ptime = 60
   stime= 200
   picks_dict={}
   picks_dict[classifier.mode]={}
   for phase in classifier.phases:
     picks_dict[classifier.mode][phase]={}
     picks_dict[classifier.mode][phase]['utc']=[]
     picks_dict[classifier.mode][phase]['value']=[]
     picks_dict[classifier.mode][phase]['pick']=[] 
     picks_dict[classifier.mode][phase]['out']=[] 
     picks_dict[classifier.mode][phase]['out_t'] =[]
 
   for count, tr in enumerate(stream[:1]):
     info = com.get_info(tr)
     pslice = tr.slice(info['stt']+ ptime+2, info['stt']+ ptime+2+classifier.win) # take a slice of required size 2 seconds into the P phase     
     pslice_proc = classifier.preproc(pslice)
     pred = classifier.get_picks(pslice_proc)
     pred_phase = classifier.phases[np.argmax(pred)]
     print(f'The predicted phase is {pred_phase}')
     picks_dict = classifier.post_proc(pslice_proc, pred, count, info['stt']+ ptime+2, info['stt'], info['f'], picks_dict) 
     print(f'picks_dict = {picks_dict}')  

if __name__ == '__main__':
   main() 