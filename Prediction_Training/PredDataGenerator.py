"""
A class to convert a SeedDataset object into processed data as CSV file
"""
import numpy as np
import obspy
from obspy.core.stream import Stream
import os
import sys
import random
sys.path.append('..\\utils\\')
import common as com
from SeedDataset import SeedDataSet
from obspy.signal.trigger import pk_baer, plot_trigger
import matplotlib.pyplot as plt
import pandas as pd

class PredDataGenerator:
   def __init__(self, dataset, channels=['HHZ'], f=100, filt=0, 
                window=4, pslim=10, psmax=100, psize=4, ssize=100, proc='log10', 
                phase_start=0, numslices=1, norm='global',
                dataset_type='train', train_test_ratio=0.95,
                plot=2, shuffle=1, num_plots=10, return_type='list', save_test =0):
     """
     dataset: SeedDataset object
 
     channels: seismic channels to select

     f: sampling rate for the generated data

     filt: denotes the filter to be used; 0: no filter; [x]: lowpass at x Hz; [x, y]: bandpass between x and y Hz

     window: size in seconds for each data sample
   
     pslim: minimum distance between P and S arrivals

     psize: seconds after arrival considered as p class

     ssize: data considered as S phase

     proc: processing for data

     phase_start: time before a phase arrival to include in corresponding class

     numslices: number of slices to take from each class for one event

     norm: the type of normalization

     train_test_ratio: The ratio of data to convert into training dataset

     plot: if plotting predefined number of data samples from (num_plots); 0: no plots; 1: only histograms are plotted; 2: data is plotted for each event

     shuffle: 0: no shuffle; 1: shuffle events 

     num_plots: number of plots to make if plot=1

     return_type: type of data returned;'list' implemented
     
     save_test: saving additional test data as traces in another folder

     return_type: the type of data to return (currently not implemented)

     save_test:
     """
     self.dataset = dataset
     self.ptime = dataset.ptime
     self.stime = dataset.stime
     self.channels = channels
     self.norm= norm
     self.f =f
     self.filt=filt
     self.window = window
     self.wlen = int(self.window * self.f)
     self.psize = psize # seconds after arrival considered as p class
     self.ssize = ssize # seconds after arrival considered as s class
     self.proc=proc
     if self.proc == '':
       self.fproc = np.array
     elif self.proc == 'log10':
       self.fproc = np.log10
     elif self.proc == 'log':
       self.fproc = np.log
     self.pslim = pslim
     self.psmax = psmax
     self.phase_start = phase_start # For p and s phases the earliest location of arrival in a window
     self.numslices=numslices # the number of slices per class from each event
     
     self.return_type= return_type
     self.dataset_type= dataset_type
     self.ratio= train_test_ratio
     self.save_test= save_test
     self.plot = plot
     self.shuffle = shuffle # 0: no shuffle; 1: shuffle data slices 2: shuffle data slices as well as streams before train test division 
     self.num_plots = num_plots
     # global params
     self.global_params ={}
     self.global_params['dmax'] = self.fproc(9000000.0)
     self.global_params['dmin'] = self.fproc(1.0)
     self.global_params['smax'] = self.fproc(9000000.0)
     self.global_params['smin'] = self.fproc(400.0)
     #self.global_params['psmax'] = 200
     self.global_params['psmax'] = self.psmax
     self.global_params['psmin'] =  self.pslim
     
     self.num_streams = 0
     self.num_slices =0
     
   def npts(self, t, f=0):
     """
     This function obtains the location in a trace based on seconds from the trace start and self.f

     t: seconds from the trace start time or list of start times

     returns integer position in the trace as a single integer or list of positions defined by t
     """
     if f==0: f = self.f
     if isinstance(t, tuple) or isinstance(t, list):
       res = []
       for item in t:
         res.append(int(item* f))
     else:
         res = int(t* f)
     return res

   def get_name(self, string):
     """
     This function obtains a string from the defined parameters for class

     string: prefix for the name

     returns a string
     """
     if isinstance(self.filt, list):
       filt_str = '_'.join(self.filt) if len(self.filt)>1 else f'{self.filt[0]}'
     else:
       filt_str = f'{self.filt}'
     chan='_'.join(self.channels)
     
     chan=chan.replace('?', 'all') 
     chan=chan.replace('*', 'all')
     
     name = f'{string}_{self.dataset_type}_{self.f}hz_{filt_str}filt_{self.window}s_{chan}channels_{self.norm}norm_{self.proc}proc'
     
     return name
  

   def get_arrivals(self, st_len):
     """
     This function computes the time of P and S arrivals based on length of current trace in seconds

     st_len: len of trace in seconds

     returns a dictionary with phase arrivals in seconds
     """        
     # location of phase arrivals          
     arrivals={}
     arrivals['p']= self.ptime # location of p arrival
     arrivals['s']= st_len-self.stime # location of s arrival
     return arrivals

   
   def get_rand_start(self, data, num):
      """
      This function provides a given number of random starting points for a given array and class parameters

      data: input data as numpy array

      num: number of starting points to provide

      returns two lists: one of integers indicating random starting points and the seconds of corressponding data slices
      """
      starts = np.random.randint(0, len(data)-self.wlen, num).tolist()
      slices = []
      for i in np.arange(len(starts)):
        sl = data[starts[i]:(starts[i]+self.wlen)]
        slices.append(sl)
      return slices, starts  

   def get_rand_start_st(self, st, num):
      """
      This function provides a given number of random starting points for a stream and class parameters

      st: input data as stream

      num: number of starting points to provide

      returns two lists: one of integers indicating random starting points and the seconds of corressponding data sliced streams
      """
      info = com.get_info(st)
      len_st = info['endt'] - info['stt']
      start_ind = np.random.uniform(0, len_st-self.window, num).tolist()
      slices = []
      starts = []
      for i in np.arange(len(start_ind)):
        start = info['stt']+start_ind[i] 
        sl = st.slice(start, start+self.window)
        slices.append(sl)
        starts.append(start)
      return slices, starts            

   
   def preproc(self, stream, to_numpy=1, squeeze=1):
     '''
     This function procesess a stream by detrending, filtering and selecting the required data from stream

     stream: stream to be processed

     to_numpy: if converted to array format

     squeeze:if extra dims to be removed

     returns an array of processed data     
     '''
     st = stream.copy()    # make a copy of stream for preprocessing     
    
     st.detrend('demean')  # remove mean
   
     info=com.get_info(stream)
     if self.f and (info['f'] != self.f): # if original frequency is different then interpolate to required frequency
       print('Stream interpolated from frequency {} to {}.'.format(info['f'], self.f))  
       st.interpolate(sampling_rate=f)  
     if self.filt: # apply filter if required
     
       if len(self.filt)==1:
         st.filter('lowpass', freq=self.filt[0])
       else: 
         st.filter('bandpass', freqmin=self.filt[0], freqmax=self.filt[1])  # optional prefiltering  
     
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
     
     if to_numpy:
       # reshape to required format
       batch_data = np.array(batch_data)
          
     return batch_data

   def plot_hist(self, data, plotpath, bins=50, label=''):
     """
      This function plots histogram of given data

      data: data as array

      plotpath: path where the plot is to be saved

      bins: number of histogram bins

      label: label used for x axis and file name

      saves the plot to the plotpath with name from label
     """
     plt.figure()
     d_max = np.amax(data)
     d_min = np.amin(data)
     plt.hist(data, bins =bins, label = 
       f'{label} ({len(data)} events) max={d_max:.4f} min={d_min:.4f}')
     plt.xlabel(label)
     plt.ylabel('Counts')
     plt.tight_layout() 
     plt.legend()
     plt.savefig(os.path.join(plotpath, f'{label}.png'))
     plt.close()

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

   def generate_dataset(self, outdir):
      """
      This function generates a data set and saves it to a directory

      outdir: directory for dataset
      """  
      stream_list = self.dataset.stream_to_list(f=[self.f], channels=self.channels)      
      if len(stream_list) == 0: # if path is empty
        print(f'No events were found in {self.dataset.path} for {self.f}Hz and channels={self.channels}')
        print(self.dataset.channels)
        print(self.dataset.freq)
        exit(0)
      else:
        print(f'{len(stream_list)} streams are found for {self.f} Hz and {self.channels} channel')
      com.safe_mkdir(outdir)        
      print(f'{outdir} folder is created.')
      if self.plot==2:
        eventspath = os.path.join(outdir, 'events') #
        com.safe_mkdir(eventspath)
        print(f'{eventspath} folder is created.')
      if self.shuffle: # if shuffling the traces
        random.seed(3)
        stream_list = random.sample(stream_list, len(stream_list))
      data_len= self.npts(self.window) # size of input window
      self.len = len(stream_list) # number of streams
      self.train_len = int(self.ratio * self.len) # number of training samples
      st_list = stream_list[:self.train_len] if self.dataset_type=='train' else stream_list[self.train_len:]
      if self.save_test: # if some test traces have to be set aside
        testpath = os.path.join(outdir, 'test_traces') #
        if os.path.exists (testpath) :
          os.system(f"rmdir /S {testpath}")
        com.safe_mkdir(testpath)
        print(f'{testpath} folder is created.')
        st_test = stream_list[self.train_len:]
        for st in st_test:
          com.save_stream(st, testpath)
        print(f'{len(st_test)} traces are saved in {testpath} folder for additional tests.')
      data_dict = {} # dictionary for dataset
      data_dict['stream_id'] =[]  
      data_dict['slice_start'] = []
      data_dict['sampling_rate'] =[]
      data_dict['channel'] =[]
      data_dict['smax'] = []
      data_dict['tmaxpos'] = []
      data_dict['data'] = [] 
      removed = 0 
      # Loop through list
      for n, st in enumerate(st_list[::]):
        remove_tr = 0
        if self.plot==2 and n<self.num_plots:
          eventdir = os.path.join(eventspath, f'event{n}')
          com.safe_mkdir(eventdir)    
        
        info = com.get_info(st)
        f_st = info['f']
          
        # trace statistics
        stt = info['stt']
        endt = info['endt']
        sta = info['sta']
        net = info['net']
        loc = info['loc']
        st_len = endt-stt

        # id for a trace
        id = com.get_id(st)
        # get class divisons          
        arrivals=self.get_arrivals(st_len)
        
        if (arrivals['s'] - arrivals['p'] < (self.pslim)) or (arrivals['s'] - arrivals['p'] > (self.psmax)):
            print(f'The required amount of data is not available thus the stream {n} is dropped')
            remove_tr = 1
            removed+=1
            continue
        if remove_tr: continue
        self.num_streams+=1
        
        pdata = st.slice(stt+ arrivals['p'], stt+ arrivals['p']+self.psize) # p phase data
        sdata =  st.slice(stt+ arrivals['s'], stt+ arrivals['s']+self.ssize) # s phase data
        # get S max
        sdata.detrend('demean')
        smax, maxpos =0, 0
        sdata =np.absolute(sdata.data)
        if max(sdata) > smax:
          smax = max(np.absolute(sdata))
          maxpos = np.argmax(sdata)
        tmaxpos = maxpos/f_st  # time in seconds for peak        
        slices, starts= self.get_rand_start_st(pdata, self.numslices)
        for i in range(self.numslices):
          data_dict['stream_id'].append(id)
          data_dict['slice_start'].append(starts[i])
          data_dict['sampling_rate'].append(self.f)
          data_dict['channel'].append('/'.join(self.channels))
          data_dict['smax'].append(self.fproc(smax)) # get processed value of s max
          t_max = (stt+arrivals['s']- starts[i]) # get the time of S arrival from start of P data
          #data_dict['tmaxpos'].append(t_max+ tmaxpos) # time in seconds between p start and s max
          data_dict['tmaxpos'].append(arrivals['s']-arrivals['p']) # time in seconds between p start and s start
          proc_data = self.preproc(slices[i]) # get p phase data in correct format
          plot = (self.plot>1) and (n<self.num_plots)
          if plot:
            self.plot_data(proc_data, eventdir, label='event_data_{i}')
          proc_data = np.absolute(proc_data)
          proc_data = np.where(proc_data<=1, 0, self.fproc(np.absolute(proc_data))) # preproc using selected function
          if plot:
            self.plot_data(proc_data, eventdir, label=f'abs_{self.proc}_{i}')
          # normalize p phase data
          if self.norm=='global':
            dmin, dmax = self.global_params['dmin'], self.global_params['dmax']
          else:
            dmin, dmax =0, 0
          proc_data =com.normalize(proc_data, norm='unity', dmin=dmin, dmax=dmax)
          if plot:
            self.plot_data(proc_data, eventdir, label='normed_{i}')    
          data_dict['data'].append(proc_data) 
       
          self.num_slices+=1
      # normalize smax
      if self.norm=='global':
       dmin, dmax = self.global_params['smin'], self.global_params['smax']
      else:
       dmin, dmax =min(data_dict['smax']), max(data_dict['smax'])
                                                             
      data_dict['smax'] = list(com.normalize(np.array(data_dict['smax']), norm='unity', dmin=dmin, dmax=dmax))
      # normalize time of smax
      if self.norm=='global':
       dmin, dmax = self.global_params['psmin'], self.global_params['psmax']
      else:
       dmin, dmax =min(data_dict['tmaxpos']), max(data_dict['tmaxpos'])
                                                             
      data_dict['tmaxpos'] = list(com.normalize(np.array(data_dict['tmaxpos']), norm='unity', dmin=dmin, dmax=dmax))
      if self.plot:
         self.plot_hist(data_dict['smax'], outdir, label='S_Max' )
         self.plot_hist(np.array(data_dict['data']).flatten(), outdir, label='data' )
         self.plot_hist(data_dict['tmaxpos'], outdir, label='t_max' )
      data_dict = com.dict_for_pd(data_dict)
      df = pd.DataFrame.from_dict(data_dict)
      df.to_csv(os.path.join(outdir, 'dataset.csv'), index=False)
      
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
   dataPath = "..\\SeedData\\PeshawarData2016_2019\\*\\*.mseed" 
   #dataPath = "..\\SeedData\\Jan2018-Jan2020_seed\\mseedFiles\\*.mseed"

   #channels=['HHE', 'HHN', 'HHZ']
   channels=['HHZ']
   #channels=['BHZ']
   f=100
   dataset_type='train'    
   dataset = SeedDataSet(dataPath, ptime=60, stime=200)      
   gendata = PredDataGenerator(dataset, dataset_type=dataset_type, channels=channels, f=f, plot=2, norm='global', proc='log')
   basepath = f'..\\Prediction_datasets\\{gendata.dataset_type}_dataset'
   datadir= os.path.join(basepath, gendata.get_name('Peshawar_psdis'))
   gendata.generate_dataset(datadir)
   config = com.get_class_config(gendata, [com.decorate('Dataset Configuration')]) 
   com.to_file(config, os.path.join(datadir, 'config.txt'))

if __name__ == '__main__':
   main() 