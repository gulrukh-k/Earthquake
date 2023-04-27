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
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model

# class defined for classifier models
class Classifier():
   def __init__(self, mode='stftcnn', f=40, win = 4, filt=[5], # params for preprocessing
                single=0, pthresh=0.95, sthresh=0.85,            # params for implementation
                sta_lta_check=0, sta_lta_p = 1.2, sta_lta_s = 1.1, sta_win=2, # additional checks
                phases =['n', 'p', 's'], full_trace=1,          # phases present in the model
                channels =['Z'], half=0, factor=1, nperf=1,                            # channel used
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
     self.half= half
     self.factor= factor
     self.nperf = nperf
     self.channels = channels
     self.multi_mode = ''
     self.mode = mode   # the mode can help select between different models
     # load model
     self.multi_mode = 0
     if 'stft' in self.mode:        
       self.preproc = self.stft_proc
       
       if model_path:
         self.model_path = model_path
       
       elif mode[-2:]=='n1':
         self.channels = ['BH?']
         self.model_path = '..\\Phase_Classification_Training\\results\\fft_filt_5hz_half_stft_4s_tails_3class_3_layer_same_adam_rd_lr_ver2_filt3_dsize64_allchannels\\saved_model\\model'
         self.f = f
         self.half=1
         self.preproc = self.stft_proc_multi
         self.multi_mode = 'most_frequent'
       elif mode[-2:]=='n2':
         self.model_path = '..\\Phase_Classification_Training\\results\\fft_filt_5hz_half_stft_4s_tails_3class_3_layer_same_adam_rd_lr_ver2_filt3_dsize64\\saved_model\\model.h5'
         self.half=1
       elif mode[-2:]=='n3':
         self.model_path = '..\\Phase_Classification_Training\\results\\peshawar_train_freq_100hz_filt_0_half_stft_4s_3class_tails_filt3_dsize64\\saved_model\\model'
         self.f = 100
         self.filt=0
         self.pthresh=0.8
         self.sthresh=0.8
         self.phases =['n', 'p', 's']
         self.half=1
       elif mode[-2:]=='n4':
         self.model_path = '..\\Phase_Classification_Training\\results\\testing\\saved_model\\model'
         self.f = 100
         self.filt=0
         self.pthresh=0.8
         self.sthresh=0.8
         self.phases =['p', 's', 'n']
         self.half=1
       elif mode[-2:]=='n5':
         self.model_path = '..\\Phase_Classification_Training\\results\\pesh_stft_0filt_hhz_mfilt3\\saved_model\\model'
         self.f = 100
         self.filt=0
         self.pthresh=0.8
         self.sthresh=0.8
         self.phases =['p', 's', 'n']
         self.half=1
       elif mode[-2:]=='n6':
         self.model_path = '..\\Phase_Classification_Training\\results\\pesh_stft_5filt_hhz_mfilt3\\saved_model\\model'
         self.f = 100
         self.filt=[5]
         self.pthresh=0.8
         self.sthresh=0.8
         self.phases =['p', 's', 'n']
         self.half=1
       elif mode[-2:]=='n7':
         self.model_path = '..\\Phase_Classification_Training\\results\\pesh_stft_0filt_hhz_mfilt3_tensorboard\\saved_model\\model'
         self.f = 100
         self.filt=0
         self.pthresh=0.8
         self.sthresh=0.8
         self.phases =['p', 's', 'n']
         self.half=1
       elif mode[-2:]=='n8':
         self.model_path = '..\\Phase_Classification_Training\\results\\Peshawar_dataset_prenorm0_0filt_hhz_mfilt3_tensorboard\\saved_model\\model'
         self.f = 100
         self.filt=0
         pthresh=0.9
         sthresh=0.9
         self.phases =['p', 's', 'n']
         self.half=1
       elif mode[-2:]=='n9':
         self.model_path = '..\\Phase_Classification_Training\\results\\Peshawar_dataset_prenorm0_0filt_hhz_mfilt3_shuffled2\\saved_model\\model'
         self.f = 100
         self.filt=0
         pthresh=0.9
         sthresh=0.9
         self.phases =['p', 's', 'n']
         self.half=1
       elif mode[-2:]=='10': # ISB ConvEQZ filt0
         self.model_path = '..\\Phase_Classification_Training\\results\\ISB_dataset_prenorm0_factor2_nperf1_0filt_hhz_mfilt3\\saved_model\\model'
         self.f = 100
         self.filt=0
         pthresh=0.8
         sthresh=0.8
         self.phases =['p', 's', 'n']
         self.half=1
       elif mode[-2:]=='11':
         self.model_path = '..\\Phase_Classification_Training\\results\\ISB_dataset_prenorm0_factor10_nperf2_0.1_1_filt_hhz_mfilt3\\saved_model\\model'
         self.f = 100
         self.filt=0
         pthresh=0.8
         sthresh=0.8
         self.phases =['p', 's', 'n']
         self.half=0
         self.factor=10
         self.nperf = 2
         self.filt=[0.1, 1]
       elif mode[-2:]=='12': # PESH ConvEQX filt0
         self.model_path = '..\\Phase_Classification_Training\\results\\PESH_train_prenorm0_factor2_nperf1_0filt_allchannels_mfilt3\\saved_model\\model'
         self.channels = ['HH?']
         self.f = 100
         self.filt=0
         pthresh=0.8
         sthresh=0.8
         self.phases =['p', 's', 'n']
         self.half=1
         self.preproc = self.stft_proc_multi
         self.multi_mode = 'most_frequent'
       elif mode[-2:]=='13': # PESH ConvEQZ filt0
         self.model_path = '..\\Phase_Classification_Training\\results\\PESH_train_prenorm0_factor2_nperf1_0filt_hhz_mfilt3\\saved_model\\model'
         self.channels = ['Z']
         self.f = 100
         self.filt=0
         pthresh=0.8
         sthresh=0.8
         self.phases =['p', 's', 'n']
         self.half=1
       elif mode[-2:]=='14':
         self.model_path = '..\\Phase_Classification_Training\\results\\PESH_train_prenorm0_factor2_nperf1_1_40filt_allchannels_mfilt3\\saved_model\\model'
         self.channels = ['HH?']
         self.f = 100
         self.filt=[1, 40]
         pthresh=0.8
         sthresh=0.8
         self.phases =['p', 's', 'n']
         self.half=1
         self.preproc = self.stft_proc_multi
         self.multi_mode = 'most_frequent'
       elif mode[-2:]=='15': # ISB ConvEQX filt1-40
         self.model_path = '..\\Phase_Classification_Training\\results\\ISB_train_prenorm0_factor2_nperf1_1_40filt_allchannels_mfilt3\\saved_model\\model'
         self.channels = ['HH?']
         self.f = 100
         self.filt=[1, 40]
         pthresh=0.8
         sthresh=0.8
         self.phases =['p', 's', 'n']
         self.half=1
         self.preproc = self.stft_proc_multi
         self.multi_mode = 'max'
       elif mode[-2:]=='16':  # ISB ConvEQX filt0
         self.model_path = '..\\Phase_Classification_Training\\results\\ISB_train_prenorm0_factor2_nperf1_0filt_allchannels_mfilt3\\saved_model\\model'
         self.channels = ['HH?']
         self.f = 100
         self.filt=0
         pthresh=0.8
         sthresh=0.8
         self.phases =['p', 's', 'n']
         self.half=1
         self.preproc = self.stft_proc_multi
         self.multi_mode = 'most_frequent'
       elif mode[-2:]=='17': # IRIS ConvEQZ
         self.model_path = '..\\Phase_Classification_Training\\results\\IRIS_stft_hhz_5filt_mfilt3\\saved_model\\model'
         self.half=1
         self.phases =['p', 's', 'n']
       elif mode[-2:]=='18': # IRIS ConvEQX
         self.channels = ['BH?']
         self.model_path = '..\\Phase_Classification_Training\\results\\IRIS_stft_allchannels_5filt_mfilt3\\saved_model\\model'
         self.f = f
         self.half=1
         self.preproc = self.stft_proc_multi
         self.multi_mode = 'most_frequent'
         self.phases =['p', 's', 'n']
       else:
         self.model_path = '..\\Phase_Classification_ANN\\results\\fft_filt_5hz_half_stft_4s_tails_3class_3_layer_same_adam_rd_lr_ver2_filt3_dsize64\\saved_model\\model.h5'
       
       try:
         self.model= load_model(self.model_path)
         self.model.summary()
         print(f'The model {self.model_path} is loaded.')
  
       except:
            
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
       if phase=='p': self.phase_lims[phase]=[-2, 5]  # p class window
       if phase=='s': self.phase_lims[phase]=[-2, 5]  # s class window
       elif phase == 'n': self.phase_lims[phase]=[0, 0]  # noise    
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

   model_path: if using a model other than predefined then provide the path and adjust other paremetrs by hand

   model_params: architectural params

   proc_level   : level for preprocessing; 'trace': performed on the whole trace; 'slice' or 'batch': performed on the current slice

   eval_level   : level for application; 'trace': performed on the whole trace; 'slice' or 'batch': performed on the current slice    
   """
   # A similar function can be created for required params
   def get_cnn_params(self, n=0):
     '''
     This function obtains parameters for some predefined models.

     n: model ID

     returns a dictionary containing architectural parameters.     
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
     This function obtains model parameters for attcnn architecture.

     n: model ID 

     returns a dictionary containing architectural parameters.     
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

   def stft_proc(self, stream, to_numpy=1):
     '''
     This function processes a stream by taking STFT

     stream: stream to be processed

     to_numpy: if the stream is to be converted to numpy

     returns the processed data.     
     '''
     
     stft_data = com.stft_proc(stream, norm=1, f=self.f, filt=self.filt, channels=self.channels, half=self.half, factor=self.factor, nperf=self.nperf, to_numpy=to_numpy, squeeze=0)
     
     return stft_data

   def stft_proc_multi(self, stream, to_numpy=1):
     '''
     This function processes a stream by taking STFT

     stream: stream to be processed

     to_numpy: if the stream is to be converted to numpy

     returns the processed data.     
     '''
     stft_data = com.stft_proc(stream, norm=1, f=self.f, filt=self.filt, channels=self.channels, half=self.half, factor=self.factor, nperf=self.nperf, to_numpy=to_numpy, squeeze=0)
     #data_shape = stft_data.shape
     stft_data = np.swapaxes(stft_data, -1, 0)
     #print(data_shape, stft_data.shape)
     #exit()
     return stft_data

   def post_proc(self, batch_data, prediction, count, tm, stt, f_act, picks_dict, triggers):
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
     prediction = np.array(prediction)
     
     if self.multi_mode == 'mean': 
       pred = np.mean(prediction, axis=-1)
       pred = np.argmax(pred)
     elif self.multi_mode == 'max': 
       pred = np.max(prediction, axis=-1)
       pred = np.argmax(pred)
     else:
       pred = np.argmax(prediction, axis=-1) 
       pred = com.most_frequent(pred)
 
     max_ind = np.argmax(prediction[:, pred])     
     if prediction[max_ind][pred] < self.peak_thresh[self.phases[pred]]:        
         pred=self.phases.index('n')  # noise if value is less than threshold      
     
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
     value = prediction[max_ind][pred] # value of the predicted phase
     
     # save the pick in the dict
     #print(picks_dict[self.mode][phase]['utc'])
     picks_dict[self.mode][phase]['utc'].append(utctime)
     picks_dict[self.mode][phase]['value'].append(value)
     picks_dict[self.mode][phase]['pick'].append(loc) # locus wrt actual trace   
     
     # save the values of all outputs
     for p_cnt, phase in enumerate(self.phases):
       picks_dict[self.mode][phase]['out'].append(prediction[:, p_cnt])  
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
     
     return prediction  

def main():
   """
   Small working example:

   # obspy is required for processing of data in seed format
   import obspy

   # select the required mode
   mode = 'stftcnn5'
   classifier = Classifier(mode= mode)
   
   # The path to data
   datapath = 'data\\test_traces\\*.mseed'

   # read data in stream
   stream = obspy.read(datapath)

   # arrival times for the trace
   ptime = 60
   stime= 200

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
   for count, tr in enumerate(stream[:1]):
     
     # get information about stream

     info = com.get_info(tr)

     # slice in required size

     pslice = tr.slice(info['stt']+ ptime+2, info['stt']+ ptime+2+classifier.win) # take a slice of required size 2 seconds into the P phase 

     # process the slice  
  
     pslice_proc = classifier.preproc(pslice)

     # get prediction from the classifier

     pred = classifier.get_picks(pslice_proc)

     # convert one hot to phase name

     pred_phase = classifier.phases[np.argmax(pred)]

     print(f'The predicted phase is {pred_phase}')
     
     # load values in picks dictionary

     picks_dict = classifier.post_proc(pslice_proc, pred, count, info['stt']+ ptime+2, info['stt'], info['f'], picks_dict)
 
     print(f'picks_dict = {picks_dict}') 
   """
   model_path = '..\\Phase_Classification_ANN\\results\\fft_filt_5hz_half_stft_4s_tails_3class_3_layer_same_adam_rd_lr_ver2_filt3_dsize64_allchannels\\saved_model\\model'

   classifier = Classifier(mode='stftcnn5')

   
if __name__ == '__main__':
   main() 